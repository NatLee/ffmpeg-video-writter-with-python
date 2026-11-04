"""The original 2020 class, kept so ``from FFmpegVideoWritter import FFmpegVideoWritter`` still works.

``FFmpegVideoWritter`` reads one video and writes another: ``get_frame()`` gives you the next
frame of ``in_filename`` (resampled to ``fps``) and ``write()`` appends a frame to
``out_filename``. It is now a thin wrapper around ``ffmpeg_video_writer.VideoReader`` and
``VideoWriter``, so it gets their frame checks and real error messages.

Compatibility decisions, kept in this one file on purpose:

* Changed in 1.0.0, approved by the maintainer: the output is stored as yuv420p, like
  ``VideoWriter``, so its .mp4 files play in browsers, QuickTime and phones. Up to 0.1.0 ffmpeg
  chose the pixel format itself, which for RGB frames meant 4:4:4 H.264 that many players refuse.
  yuv420p needs an even width and height, so odd-sized frames now get one black row/column.
* Still as in 2020: the encoder is the one ffmpeg picks for the file extension (H.264 for .mp4,
  mpeg4 for .avi, VP9 for .webm), with the original ``-qscale:v 0.01`` quality option.
* Bugs fixed on the way: the output now has the requested frame rate (it used to be written at
  ffmpeg's default 25 fps, so the video played too slowly), the last partial second of the input
  is no longer cut off, ``close()`` no longer hangs when frames were left unread,
  ``output_to_gif()`` works without an argument, gray input works, and sources without a frame
  count (mkv, webm) open.
* ``output_to_gif()`` now builds a colour palette from the video first (``to_gif``): cleaner
  colours and roughly half the file size of ffmpeg's plain GIF conversion.

This file can go once nobody imports ``FFmpegVideoWritter`` any more.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ffmpeg_video_writer import VideoReader, VideoWriter, to_gif

#: The only pipe format that carries numpy frames.
PIPE_FORMAT = "rawvideo"

#: The 2020 quality option. libx264 ignores it; mpeg4-family encoders (.avi) read it as
#: "best quality", and their output would turn blurry without it.
LEGACY_OUTPUT_ARGS = ("-qscale:v", "0.01")


class FFmpegVideoWritter:
    """FFmpeg video writter

    Attributes
    ----------
        in_filename: str
            your input file name
        out_filename: str
            your output file name
        video_format: str
            pipe format, must be 'rawvideo'
        input_pix_fmt: str
            frame layout: 'rgb24' (default), 'bgr24', 'rgba', 'bgra' or 'gray'
        input_width, input_height: int
            size of the input frames
        input_frames: int
            number of frames in the input
        input_length: int
            input length in whole seconds
        output_frames: int
            about how many frames get_frame() returns at the requested fps
    Methods
    ----------
        get_frame
            return a frame from video, or None at the end
        write(frame=np.ndarray)
            write a frame into file
        close
            close the session
        output_to_gif
            get a gif output
    """

    def __init__(
        self,
        in_filename: str,
        out_filename: str,
        fps: float = 27.0,
        video_format: str = PIPE_FORMAT,
        input_pix_fmt: str = "rgb24",
    ) -> None:
        if video_format != PIPE_FORMAT:
            raise ValueError(
                f"video_format must be {PIPE_FORMAT!r}: numpy frames travel through the pipe as"
                f" raw pixels (got {video_format!r})"
            )
        self.in_filename = in_filename
        self.out_filename = out_filename
        self.video_format = video_format
        self.input_pix_fmt = input_pix_fmt

        self._reader = VideoReader(in_filename, fps=fps, pixel_format=input_pix_fmt)
        info = self._reader.info
        self.input_width = info.width
        self.input_height = info.height
        self.input_frames = info.frames or 0
        self.input_length = int(self.input_frames / info.fps) if info.fps else 0
        self.output_frames = self._reader.frame_count or 0

        # yuv420p and even padding come from VideoWriter's defaults (see the module docstring).
        self._writer = VideoWriter(
            out_filename,
            fps=fps,
            pixel_format=input_pix_fmt,
            codec=None,
            output_args=LEGACY_OUTPUT_ARGS,
        )

    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from video"""
        return self._reader.read()

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame"""
        self._writer.write(frame)
        return True

    def close(self) -> None:
        """Release the resource"""
        try:
            self._writer.close()
        finally:
            self._reader.close()

    def output_to_gif(self, gif_filename: Optional[str] = None) -> bool:
        """Get a gif output (default name: the output file name + '.gif')"""
        to_gif(self.out_filename, gif_filename or self.out_filename + ".gif")
        return True

    def __enter__(self) -> "FFmpegVideoWritter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self._writer.__exit__(exc_type, exc, tb)
        finally:
            self._reader.close()
