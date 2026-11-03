"""VideoReader, probe() and to_gif()."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from conftest import SAMPLE_FPS, SAMPLE_SECONDS, SAMPLE_SIZE, ffmpeg_lavfi, rotate_90, video_stream

from ffmpeg_video_writer import FFmpegError, FFmpegNotFoundError, VideoReader, probe, to_gif

WIDTH, HEIGHT = SAMPLE_SIZE
SAMPLE_FRAMES = SAMPLE_FPS * SAMPLE_SECONDS


def finishes_within(seconds, action) -> bool:
    worker = threading.Thread(target=action, daemon=True)
    worker.start()
    worker.join(seconds)
    return not worker.is_alive()


def test_reads_every_frame_as_writable_rgb(sample):
    with VideoReader(sample) as reader:
        frames = list(reader)
    assert len(frames) == SAMPLE_FRAMES
    assert frames[0].shape == (HEIGHT, WIDTH, 3) and frames[0].dtype == np.uint8
    frames[0][0, 0] = 0  # drawing on a frame must not raise "assignment destination is read-only"
    assert reader.read() is None  # stays at the end


def test_resamples_to_the_requested_fps(sample):
    with VideoReader(sample, fps=10) as reader:
        count = sum(1 for _ in reader)
        assert reader.frame_count == SAMPLE_SECONDS * 10
    assert count == SAMPLE_SECONDS * 10


@pytest.mark.parametrize("pixel_format, shape", [("gray", (HEIGHT, WIDTH)), ("bgra", (HEIGHT, WIDTH, 4))])
def test_other_pixel_formats(sample, pixel_format, shape):
    with VideoReader(sample, pixel_format=pixel_format) as reader:
        assert reader.read().shape == shape


def test_bgr_is_rgb_reversed(sample):
    with VideoReader(sample) as rgb, VideoReader(sample, pixel_format="bgr24") as bgr:
        np.testing.assert_array_equal(rgb.read(), bgr.read()[..., ::-1])


def test_early_close_does_not_hang(sample):
    reader = VideoReader(sample)
    reader.read()  # the rest of the frames are still waiting in the pipe
    assert finishes_within(10, reader.close)


def test_container_without_frame_count(sample, tmp_path):
    mkv = ffmpeg_lavfi(tmp_path / "sample.mkv", "-i", str(sample), "-c", "copy")
    assert "nb_frames" not in video_stream(mkv)
    with VideoReader(mkv) as reader:
        # Estimated from the duration, which in mkv also covers the audio track: close, not exact.
        assert abs(reader.info.frames - SAMPLE_FRAMES) <= 1
        assert sum(1 for _ in reader) == SAMPLE_FRAMES


def test_rotated_phone_video_has_upright_size(sample, tmp_path):
    rotated = rotate_90(sample, tmp_path / "rotated.mp4")
    with VideoReader(rotated) as reader:
        assert (reader.width, reader.height) == (HEIGHT, WIDTH)
        assert reader.read().shape == (WIDTH, HEIGHT, 3)


def test_probe(sample):
    info = probe(sample)
    assert (info.width, info.height, info.fps, info.frames) == (WIDTH, HEIGHT, SAMPLE_FPS, SAMPLE_FRAMES)
    assert info.duration == pytest.approx(SAMPLE_SECONDS, abs=0.05)
    assert (info.codec, info.pixel_format, info.has_audio) == ("h264", "yuv420p", True)


def test_missing_file(tmp_path):
    with pytest.raises(FFmpegError, match=r"No such file or directory"):
        VideoReader(tmp_path / "nope.mp4")


def test_file_that_is_not_a_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(FFmpegError, match=r"Invalid data"):
        VideoReader(junk)


def test_audio_only_file(tmp_path):
    audio = ffmpeg_lavfi(tmp_path / "tone.m4a", "-f", "lavfi", "-i", "sine=duration=1")
    with pytest.raises(ValueError, match=r"has no video stream"):
        VideoReader(audio)


def test_half_downloaded_mp4_is_reported(sample, tmp_path):
    # ffmpeg writes the mp4 index ("moov atom") at the end, so a cut-off file cannot be opened.
    broken = tmp_path / "broken.mp4"
    data = sample.read_bytes()
    broken.write_bytes(data[: len(data) // 2])
    with pytest.raises(FFmpegError, match=r"moov atom not found"):
        VideoReader(broken)


def test_missing_ffmpeg(sample):
    with pytest.raises(FFmpegNotFoundError, match=r"winget install ffmpeg"):
        VideoReader(sample, ffmpeg="no-such-ffmpeg")


def test_to_gif(sample, tmp_path):
    gif = to_gif(sample, tmp_path / "out.gif", fps=5, width=160)
    stream = video_stream(gif)
    assert stream["codec_name"] == "gif"
    assert (stream["width"], stream["height"]) == (160, 120)
    assert int(stream["nb_read_frames"]) == SAMPLE_SECONDS * 5


def test_to_gif_reports_ffmpeg_errors(tmp_path):
    with pytest.raises(FFmpegError, match=r"(?s)Converting .* to GIF failed .*No such file"):
        to_gif(tmp_path / "missing.mp4", tmp_path / "out.gif")
