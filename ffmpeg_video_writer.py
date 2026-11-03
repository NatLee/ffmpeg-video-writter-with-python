"""Write numpy frames to a video with ffmpeg, and get a real error when something is wrong.

Piping raw frames into ffmpeg is fast and flexible, but every mistake looks the same from
Python: ffmpeg quits, the pipe breaks, and you are left with an empty or garbled file and no
explanation. This module checks every frame before it reaches ffmpeg and turns ffmpeg's own
error message into a Python exception.

    from ffmpeg_video_writer import VideoWriter

    with VideoWriter("out.mp4", fps=30) as video:
        for frame in frames:          # numpy arrays, height x width x 3, uint8, RGB
            video.write(frame)

Public API: ``VideoWriter``, ``VideoReader``, ``probe``, ``to_gif`` and the exceptions
``FFmpegError``, ``FFmpegNotFoundError`` and ``FrameError``.
"""

from __future__ import annotations

import collections
import json
import os
import shlex
import shutil
import subprocess
import threading
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "FFmpegError",
    "FFmpegNotFoundError",
    "FrameError",
    "PIXEL_FORMATS",
    "VideoInfo",
    "VideoReader",
    "VideoWriter",
    "probe",
    "to_gif",
]

PathLike = Union[str, "os.PathLike[str]"]
Rate = Union[float, int, Fraction, str]

#: Frame layouts that can travel through the pipe, as ffmpeg pixel format -> channels per pixel.
#: Every one of them uses one byte per channel, which is why frames must hold uint8 values.
PIXEL_FORMATS = {"rgb24": 3, "bgr24": 3, "rgba": 4, "bgra": 4, "gray": 1}

#: Flags for every ffmpeg/ffprobe call: no banner and only real errors in the log, so the log
#: that ends up in an exception is the part that explains the failure.
LOG_ARGS = ("-hide_banner", "-loglevel", "error")
#: ffmpeg also prints a progress line unless told not to (ffprobe has no such option).
FFMPEG_ARGS = (*LOG_ARGS, "-nostats")

#: How many of ffmpeg's last log lines are kept for the exception message.
LOG_TAIL_LINES = 40

#: How long to wait for ffmpeg to exit after its pipe broke, before killing it.
EXIT_TIMEOUT_SECONDS = 10

#: GIF filter: build a 256-colour palette from the video itself, then map frames onto it with
#: ordered (Bayer) dithering and redraw only the part that changed. On the examples this is about
#: half the size of ffmpeg's default error-diffusion dithering.
GIF_PALETTE_FILTER = (
    "split[a][b];[a]palettegen[palette];"
    "[b][palette]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
)

INSTALL_HINT = (
    "Install ffmpeg (it includes ffprobe) and make sure it is on your PATH, "
    "or pass the full path with ffmpeg='/path/to/ffmpeg'. "
    "Debian/Ubuntu: sudo apt install ffmpeg | macOS: brew install ffmpeg | "
    "Windows: winget install ffmpeg | conda: conda install -c conda-forge ffmpeg"
)


# --------------------------------------------------------------------------- errors


class FFmpegError(RuntimeError):
    """ffmpeg (or ffprobe) failed. The message ends with what ffmpeg itself printed.

    Attributes:
        cmd: the command line that was run.
        returncode: ffmpeg's exit code, or None if it never ran.
        stderr: the last lines ffmpeg wrote to its error log.
    """

    def __init__(
        self,
        message: str,
        *,
        cmd: Sequence[str] = (),
        returncode: Optional[int] = None,
        stderr: str = "",
    ) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        parts = [message]
        if stderr:
            parts.append("ffmpeg said:\n" + _indent(stderr))
        elif returncode is not None:
            parts.append("ffmpeg printed no error message.")
        if self.cmd:
            parts.append("command: " + shlex.join(self.cmd))
        super().__init__("\n".join(parts))


class FFmpegNotFoundError(FFmpegError):
    """The ffmpeg or ffprobe executable could not be found."""


class FrameError(ValueError):
    """A frame does not fit the video: wrong shape, size, channel count or dtype.

    The frame is rejected before it reaches ffmpeg, so the video written so far is intact and
    you can fix the frame and keep writing.
    """


# --------------------------------------------------------------------------- ffmpeg plumbing


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


def _find_executable(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise FFmpegNotFoundError(f"Cannot find {name!r}. {INSTALL_HINT}")
    return found


def _ffprobe_next_to(ffmpeg: str) -> str:
    """ffprobe ships with ffmpeg: when ffmpeg is given as a path, look in the same folder."""
    folder = os.path.dirname(ffmpeg)
    return os.path.join(folder, "ffprobe") if folder else "ffprobe"


def _parse_rate(value: Rate, name: str) -> Fraction:
    """Accept 30, 29.97, Fraction(30000, 1001) or "30000/1001"; reject zero and negatives."""
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{name} must be a positive number, got {value!r}") from None
    if rate <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return rate


def _check_pixel_format(pixel_format: str) -> int:
    if pixel_format not in PIXEL_FORMATS:
        raise ValueError(
            f"pixel_format must be one of {sorted(PIXEL_FORMATS)}, got {pixel_format!r}"
        )
    return PIXEL_FORMATS[pixel_format]


class _Process:
    """A running ffmpeg whose error log is read in the background.

    ffmpeg writes its log to stderr. If nobody reads that pipe it fills up (64 KiB on Linux,
    less on Windows) and ffmpeg stops forever, waiting for room. A daemon thread keeps the pipe
    empty and remembers the last lines, so they can go into the exception if ffmpeg fails.
    """

    def __init__(self, cmd: List[str], *, stdin=None, stdout=None) -> None:
        self.cmd = cmd
        try:
            self.popen = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:  # removed between shutil.which() and here
            raise FFmpegNotFoundError(f"Cannot start {cmd[0]!r}. {INSTALL_HINT}") from exc
        self._log: collections.deque = collections.deque(maxlen=LOG_TAIL_LINES)
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()

    def _drain_stderr(self) -> None:
        for raw in self.popen.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._log.append(line)
        self.popen.stderr.close()

    def wait(self, timeout: Optional[float] = None) -> int:
        try:
            code = self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            code = self.popen.wait()
        self._reader.join()
        return code

    def error(self, message: str) -> FFmpegError:
        return FFmpegError(
            message, cmd=self.cmd, returncode=self.popen.returncode, stderr="\n".join(self._log)
        )


def _run(cmd: List[str], what: str) -> bytes:
    """Run ffmpeg/ffprobe to completion; return stdout or raise FFmpegError with its log."""
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(f"Cannot start {cmd[0]!r}. {INSTALL_HINT}") from exc
    if done.returncode != 0:
        log = done.stderr.decode("utf-8", "replace").strip().splitlines()[-LOG_TAIL_LINES:]
        raise FFmpegError(
            f"{what} failed (exit code {done.returncode}).",
            cmd=cmd,
            returncode=done.returncode,
            stderr="\n".join(log),
        )
    return done.stdout


# --------------------------------------------------------------------------- frame checks


def _suggest_pixel_format(channels: int) -> str:
    matches = [name for name, count in PIXEL_FORMATS.items() if count == channels]
    if matches:
        return f" For {channels}-channel frames use pixel_format={matches[0]!r}."
    return ""


def _to_uint8(frame: np.ndarray, index: int) -> np.ndarray:
    """Convert to uint8 only if no pixel value changes; otherwise explain what is wrong.

    ``astype(np.uint8)`` never complains: 0..1 floats become a black video and 300 wraps
    around to 44. So a non-uint8 frame is accepted only when every value is already a whole
    number from 0 to 255.
    """
    if frame.dtype.kind not in "uif":
        raise FrameError(
            f"frame {index} has dtype {frame.dtype}; frames must hold numbers 0-255 (uint8)."
            + (" Multiply a mask by 255 first." if frame.dtype.kind == "b" else "")
        )
    is_float = frame.dtype.kind == "f"
    if is_float and not np.isfinite(frame).all():
        problem = "contains NaN or infinity"
        fix = "np.nan_to_num(frame)"
    else:
        low, high = frame.min(), frame.max()
        whole = not is_float or not np.mod(frame, 1).any()
        if 0 <= low and high <= 255 and whole:
            return frame.astype(np.uint8)
        if is_float and 0 <= low and high <= 1:
            problem = f"looks scaled to 0..1 (min {low:g}, max {high:g}); uint8 would make it black"
            fix = "(frame * 255).round().astype(np.uint8)"
        elif low < 0 or high > 255:
            problem = f"has values outside 0..255 (min {low:g}, max {high:g}) that would wrap around"
            fix = "np.clip(frame, 0, 255).astype(np.uint8)"
        else:
            problem = "has fractional values that would be cut off"
            fix = "frame.round().astype(np.uint8)"
    raise FrameError(f"frame {index} has dtype {frame.dtype} and {problem}. Convert it with {fix}.")


def _check_frame(
    frame, pixel_format: str, size: Optional[Tuple[int, int]], size_source: str, index: int
) -> np.ndarray:
    """Return the frame as a C-contiguous uint8 array of shape (height, width, channels)."""
    channels = PIXEL_FORMATS[pixel_format]
    array = np.asarray(frame)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise FrameError(
            f"frame {index} has shape {array.shape}; expected (height, width, {channels})"
            f" for pixel_format={pixel_format!r}."
        )
    height, width, got = array.shape
    if got != channels:
        raise FrameError(
            f"frame {index} has {got} channel(s) but pixel_format={pixel_format!r} expects"
            f" {channels}." + _suggest_pixel_format(got)
        )
    if width == 0 or height == 0:
        raise FrameError(f"frame {index} is empty (shape {array.shape}).")
    if size is not None and (width, height) != size:
        raise FrameError(
            f"frame {index} is {width}x{height} but this video is {size[0]}x{size[1]}"
            f" ({size_source}). Every frame must have the same size; resize or crop it to"
            f" {size[0]}x{size[1]} first."
        )
    if array.dtype != np.uint8:
        array = _to_uint8(array, index)
    return np.ascontiguousarray(array)


# --------------------------------------------------------------------------- probe


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe reports about the first video stream of a file.

    ``width`` and ``height`` are the size frames are decoded at, which already accounts for
    rotation metadata (phone videos). ``fps``, ``duration`` and ``frames`` are None when the
    file does not say; ``frames`` is estimated from the duration when the container has no count.
    """

    width: int
    height: int
    fps: Optional[float]
    duration: Optional[float]
    frames: Optional[int]
    codec: str
    pixel_format: Optional[str]
    has_audio: bool


def _stream_rate(stream: dict) -> Optional[Fraction]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(stream.get(key, ""))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: PathLike, *, ffmpeg: str = "ffmpeg") -> VideoInfo:
    """Read size, frame rate, duration, codec and pixel format of a video with ffprobe."""
    path = os.fspath(path)
    ffprobe = _find_executable(_ffprobe_next_to(ffmpeg))
    output = _run(
        [ffprobe, *LOG_ARGS, "-show_streams", "-show_format", "-of", "json", path],
        f"ffprobe could not read {path}",
    )
    data = json.loads(output or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path} has no video stream.")

    width, height = int(video["width"]), int(video["height"])
    rotation = next(
        (int(side["rotation"]) for side in video.get("side_data_list", []) if "rotation" in side), 0
    )
    if rotation % 180 == 90:  # ffmpeg turns the frames upright when decoding; swap like it does
        width, height = height, width

    rate = _stream_rate(video)
    duration = _as_float(video.get("duration")) or _as_float(data.get("format", {}).get("duration"))
    frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None
    if frames is None and duration is not None and rate is not None:
        frames = round(duration * rate)
    return VideoInfo(
        width=width,
        height=height,
        fps=float(rate) if rate is not None else None,
        duration=duration,
        frames=frames,
        codec=video.get("codec_name", ""),
        pixel_format=video.get("pix_fmt"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


# --------------------------------------------------------------------------- writer


class VideoWriter:
    """Write numpy frames to a video file through an ffmpeg pipe.

    Every frame is checked before it is sent: it must be ``height x width x channels`` (or
    ``height x width`` for gray), hold 0-255 values, and have the same size as the first frame.
    A bad frame raises ``FrameError`` and nothing is sent, so the video stays intact. If ffmpeg
    itself fails, the next ``write()`` or ``close()`` raises ``FFmpegError`` with ffmpeg's message.

    The defaults make a file that plays everywhere (browsers, phones, QuickTime): H.264 with
    yuv420p colour, padded by one black pixel when the width or height is odd, because yuv420p
    needs even sizes.

    Args:
        path: output file. The extension picks the container (.mp4, .mkv, .mov, .webm, ...).
        fps: frames per second of the output (30, 29.97 or "30000/1001").
        size: (width, height) that every frame must have. Default: taken from the first frame.
        pixel_format: layout of the frames you pass: "rgb24" (numpy, PIL, imageio),
            "bgr24" (OpenCV), "rgba", "bgra" or "gray".
        codec: video encoder, e.g. "libx264", "libx265", "libvpx-vp9". None lets ffmpeg pick
            the default for the file extension.
        crf: constant quality for x264/x265/VP9/AV1. Lower is better and bigger (x264: 18 looks
            lossless, 23 is its default, 28 is small). None keeps the encoder's default.
        bitrate: target bitrate like "4M" or 800_000. Used instead of crf when set.
        output_pixel_format: pixel format stored in the file. "yuv420p" plays everywhere;
            None lets ffmpeg choose (for RGB input it often picks 4:4:4, which many players reject).
        pad_to_even: add one black row/column when the height/width is odd. Needed by yuv420p.
        audio: optional file whose first audio track is copied into the video (re-encoded to the
            container's default audio codec). The output ends with the shorter of the two.
        output_args: extra ffmpeg output options, e.g. ["-preset", "slow", "-movflags", "+faststart"].
        ffmpeg: name or path of the ffmpeg executable.
    """

    def __init__(
        self,
        path: PathLike,
        fps: Rate = 30,
        *,
        size: Optional[Tuple[int, int]] = None,
        pixel_format: str = "rgb24",
        codec: Optional[str] = "libx264",
        crf: Optional[float] = None,
        bitrate: Union[str, int, None] = None,
        output_pixel_format: Optional[str] = "yuv420p",
        pad_to_even: bool = True,
        audio: Optional[PathLike] = None,
        output_args: Sequence[str] = (),
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.path = os.fspath(path)
        self.fps = fps
        _parse_rate(fps, "fps")
        _check_pixel_format(pixel_format)
        if size is not None:
            width, height = (int(n) for n in size)
            if width <= 0 or height <= 0:
                raise ValueError(f"size must be two positive numbers (width, height), got {size!r}")
            size = (width, height)
        self.pixel_format = pixel_format
        self.codec = codec
        self.crf = crf
        self.bitrate = bitrate
        self.output_pixel_format = output_pixel_format
        self.pad_to_even = pad_to_even
        self.audio = os.fspath(audio) if audio is not None else None
        self.output_args = list(output_args)
        self._ffmpeg = _find_executable(ffmpeg)
        self._size = size
        self._size_source = "set by size=" if size is not None else "set by the first frame"
        self._process: Optional[_Process] = None
        self._frames = 0
        self._closed = False

    # -- information ----------------------------------------------------------------------

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """(width, height) every frame must have; None until the first frame or size=."""
        return self._size

    @property
    def output_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) stored in the file, after padding odd sizes to even."""
        if self._size is None:
            return None
        width, height = self._size
        if self.pad_to_even:
            width, height = width + width % 2, height + height % 2
        return (width, height)

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def command(self) -> Optional[List[str]]:
        """The ffmpeg command line, once the first frame has started ffmpeg."""
        return list(self._process.cmd) if self._process else None

    # -- writing --------------------------------------------------------------------------

    def _build_command(self) -> List[str]:
        width, height = self._size
        cmd = [self._ffmpeg, *FFMPEG_ARGS, "-y",
               "-f", "rawvideo", "-pix_fmt", self.pixel_format, "-s", f"{width}x{height}",
               "-framerate", str(self.fps), "-i", "pipe:0"]  # fmt: skip
        if self.audio is not None:
            cmd += ["-i", self.audio, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        if self.output_size != self._size:
            cmd += ["-vf", "pad={}:{}".format(*self.output_size)]
        options = {
            "-c:v": self.codec,
            "-pix_fmt": self.output_pixel_format,
            "-b:v": self.bitrate,
            "-crf": self.crf if self.bitrate is None else None,
        }
        for flag, value in options.items():
            if value is not None:
                cmd += [flag, str(value)]
        return cmd + [*self.output_args, self.path]

    def write(self, frame) -> None:
        """Check one frame and send it to ffmpeg."""
        if self._closed:
            raise ValueError(f"write() called after close() on {self.path}")
        array = _check_frame(frame, self.pixel_format, self._size, self._size_source, self._frames)
        if self._process is None:
            height, width = array.shape[:2]
            self._size = self._size or (width, height)
            self._process = _Process(self._build_command(), stdin=subprocess.PIPE)
        popen = self._process.popen
        if popen.poll() is not None:
            raise self._fail(f"ffmpeg exited before frame {self._frames} could be written")
        try:
            popen.stdin.write(memoryview(array).cast("B"))
        except OSError as exc:  # BrokenPipeError on Linux/macOS, EINVAL on Windows
            raise self._fail(f"ffmpeg stopped reading at frame {self._frames}") from exc
        self._frames += 1

    def _fail(self, message: str) -> FFmpegError:
        """ffmpeg is gone or going: collect its exit code and log, and close this writer."""
        self._closed = True
        self._close_stdin()
        code = self._process.wait(EXIT_TIMEOUT_SECONDS)
        return self._process.error(f"{message} (exit code {code}, writing {self.path}).")

    def _close_stdin(self) -> None:
        try:
            self._process.popen.stdin.close()
        except OSError:
            pass  # ffmpeg is already gone; its exit code and log say why

    def close(self) -> None:
        """Finish the file. Raises FFmpegError if ffmpeg did not exit cleanly."""
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            warnings.warn(
                f"No frames were written, so {self.path} was not created.", RuntimeWarning, stacklevel=2
            )
            return
        self._close_stdin()
        code = self._process.wait()
        if code != 0:
            raise self._process.error(
                f"ffmpeg failed while finishing {self.path} after {self._frames} frame(s)"
                f" (exit code {code})."
            )

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            # Your code raised: finish what was written so far, but let your exception win.
            self._closed = True
            if self._process is not None:
                self._close_stdin()
                self._process.wait()

    def __repr__(self) -> str:
        return f"<VideoWriter {self.path!r} {self._frames} frame(s){' closed' if self._closed else ''}>"


# --------------------------------------------------------------------------- reader


class VideoReader:
    """Read the frames of a video file as numpy arrays.

    Frames come out as ``height x width x channels`` uint8 arrays (``height x width`` for
    "gray"). Each frame is a fresh, writable array, so you can draw on it directly.

    Args:
        path: input video.
        fps: resample to this many frames per second (drops or repeats frames). None keeps the
            source rate.
        pixel_format: "rgb24", "bgr24" (OpenCV order), "rgba", "bgra" or "gray".
        ffmpeg: name or path of ffmpeg; ffprobe is looked up next to it.
    """

    def __init__(
        self,
        path: PathLike,
        fps: Optional[Rate] = None,
        *,
        pixel_format: str = "rgb24",
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.path = os.fspath(path)
        channels = _check_pixel_format(pixel_format)
        rate = _parse_rate(fps, "fps") if fps is not None else None
        executable = _find_executable(ffmpeg)
        self.info = probe(self.path, ffmpeg=ffmpeg)
        self.width, self.height = self.info.width, self.info.height
        self.pixel_format = pixel_format
        self.fps = float(rate) if rate is not None else self.info.fps
        if rate is None:
            self.frame_count = self.info.frames
        elif self.info.duration is not None:
            self.frame_count = round(self.info.duration * rate)
        else:
            self.frame_count = None
        self._shape = (self.height, self.width) if channels == 1 else (self.height, self.width, channels)
        self._frame_bytes = self.width * self.height * channels
        cmd = [executable, *FFMPEG_ARGS, "-i", self.path, "-map", "0:v:0"]
        if rate is None:
            # Every decoded frame exactly once. Without this ffmpeg makes the raw stream constant
            # rate and may repeat or drop frames where the timestamps are uneven. (ffmpeg 5.1+)
            cmd += ["-fps_mode", "passthrough"]
        else:
            cmd += ["-vf", f"fps=fps={fps}:round=up"]
        cmd += ["-f", "rawvideo", "-pix_fmt", pixel_format, "pipe:1"]
        self._process = _Process(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        self._frames = 0
        self._done = False

    @property
    def frames_read(self) -> int:
        return self._frames

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame, or None at the end of the video."""
        if self._done:
            return None
        buffer = bytearray(self._frame_bytes)
        got = self._process.popen.stdout.readinto(buffer)
        if got == self._frame_bytes:
            self._frames += 1
            return np.frombuffer(buffer, np.uint8).reshape(self._shape)
        self._done = True
        code = self._process.wait()
        if code != 0:
            raise self._process.error(
                f"ffmpeg failed while decoding {self.path} after {self._frames} frame(s)"
                f" (exit code {code})."
            )
        if got:
            raise self._process.error(
                f"the last frame of {self.path} was cut short ({got} of {self._frame_bytes} bytes);"
                f" the decoded size is not the {self.width}x{self.height} ffprobe reported."
            )
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        """Stop decoding. Safe to call before the end: ffmpeg is stopped instead of waited for."""
        if self._process.popen.poll() is None:
            # Unread frames are still in the pipe and ffmpeg is blocked writing more; waiting
            # for it would hang forever. Nothing it produces from here on is needed.
            self._process.popen.terminate()
        self._process.popen.stdout.close()
        self._process.wait()
        self._done = True

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<VideoReader {self.path!r} {self.width}x{self.height} @ {self.fps} fps>"


# --------------------------------------------------------------------------- gif


def to_gif(
    src: PathLike,
    dst: PathLike,
    *,
    fps: Optional[Rate] = None,
    width: Optional[int] = None,
    ffmpeg: str = "ffmpeg",
) -> str:
    """Convert a video to an animated GIF with a palette made for it (much better colours).

    Args:
        fps: frames per second of the GIF; lower makes a smaller file. None keeps the source rate.
        width: scale to this width, keeping the aspect ratio. None keeps the size.
    Returns:
        The GIF path.
    """
    src, dst = os.fspath(src), os.fspath(dst)
    executable = _find_executable(ffmpeg)
    steps = []
    if fps is not None:
        steps.append(f"fps={_parse_rate(fps, 'fps')}")
    if width is not None:
        steps.append(f"scale={int(width)}:-1:flags=lanczos")
    steps.append(GIF_PALETTE_FILTER)
    _run(
        [executable, *FFMPEG_ARGS, "-y", "-i", src, "-filter_complex", ",".join(steps), dst],
        f"Converting {src} to GIF",
    )
    return dst
