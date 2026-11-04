"""The 2020 API: ``from FFmpegVideoWritter import FFmpegVideoWritter`` keeps working, minus its bugs."""

from __future__ import annotations

import os
import threading
from fractions import Fraction

import numpy as np
import pytest
from conftest import SAMPLE_FPS, SAMPLE_SECONDS, SAMPLE_SIZE, ffmpeg_lavfi, video_stream

from FFmpegVideoWritter import FFmpegVideoWritter
from ffmpeg_video_writer import FFmpegError, FrameError

WIDTH, HEIGHT = SAMPLE_SIZE


def copy_all(ffvw) -> int:
    """The loop from the original README."""
    count = 0
    while True:
        frame = ffvw.get_frame()
        if frame is None:
            break
        ffvw.write(frame)
        count += 1
    ffvw.close()
    return count


def test_original_readme_loop(sample, tmp_path):
    out = str(tmp_path / "result.mp4")
    ffvw = FFmpegVideoWritter(str(sample), out)
    count = copy_all(ffvw)

    stream = video_stream(out)
    # Regression: the output used to be written at ffmpeg's default 25 fps although frames were
    # sampled at 27, and the input was cut to whole seconds (2 s source -> 54 frames, 2.16 s).
    assert Fraction(stream["r_frame_rate"]) == 27
    assert count == int(stream["nb_read_frames"]) == SAMPLE_SECONDS * 27
    assert float(stream["duration"]) == pytest.approx(SAMPLE_SECONDS, abs=0.05)
    assert (stream["width"], stream["height"]) == (WIDTH, HEIGHT)


def test_attributes(sample, tmp_path):
    ffvw = FFmpegVideoWritter(str(sample), str(tmp_path / "out.mp4"), fps=10)
    assert (ffvw.input_width, ffvw.input_height) == (WIDTH, HEIGHT)
    assert ffvw.input_frames == SAMPLE_FPS * SAMPLE_SECONDS
    assert ffvw.input_length == SAMPLE_SECONDS
    assert ffvw.output_frames == SAMPLE_SECONDS * 10
    assert (ffvw.in_filename, ffvw.video_format, ffvw.input_pix_fmt) == (str(sample), "rawvideo", "rgb24")
    copy_all(ffvw)


def test_mp4_is_h264_yuv420p_and_odd_sizes_are_padded(sample, tmp_path):
    # Default change in 1.0.0, approved by the maintainer: up to 0.1.0 this was 4:4:4 H.264
    # ("High 4:4:4 Predictive", rejected by browsers) and odd sizes were kept as they were.
    out = str(tmp_path / "odd.mp4")
    ffvw = FFmpegVideoWritter(str(sample), out)
    for _ in range(3):
        ffvw.write(ffvw.get_frame()[:-1, :-1])  # 319x239
    ffvw.close()
    stream = video_stream(out)
    assert (stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p")
    assert stream["profile"] != "High 4:4:4 Predictive"
    assert (stream["width"], stream["height"]) == (WIDTH, HEIGHT)  # 319x239 padded to 320x240


@pytest.mark.parametrize("extension, codec", [(".avi", "mpeg4"), (".webm", "vp9")])
def test_encoder_still_follows_the_extension(sample, tmp_path, extension, codec):
    out = str(tmp_path / f"out{extension}")
    ffvw = FFmpegVideoWritter(str(sample), out, fps=5)
    copy_all(ffvw)
    stream = video_stream(out)
    assert (stream["codec_name"], stream["pix_fmt"]) == (codec, "yuv420p")


def test_output_to_gif_default_name(sample, tmp_path):
    out = str(tmp_path / "result.mp4")
    ffvw = FFmpegVideoWritter(str(sample), out, fps=5)
    copy_all(ffvw)
    assert ffvw.output_to_gif() is True  # used to raise TypeError: missing 'gif_filename'
    assert video_stream(out + ".gif")["codec_name"] == "gif"
    assert ffvw.output_to_gif(str(tmp_path / "named.gif")) is True
    assert os.path.exists(tmp_path / "named.gif")


def test_close_before_reading_everything_does_not_hang(sample, tmp_path):
    ffvw = FFmpegVideoWritter(str(sample), str(tmp_path / "out.mp4"))
    ffvw.write(ffvw.get_frame())
    worker = threading.Thread(target=ffvw.close, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive()


def test_gray_input(sample, tmp_path):
    ffvw = FFmpegVideoWritter(str(sample), str(tmp_path / "gray.mp4"), input_pix_fmt="gray")
    assert ffvw.get_frame().shape == (HEIGHT, WIDTH)  # used to be a garbled (240, 320, 3)
    copy_all(ffvw)


def test_mkv_input(sample, tmp_path):
    mkv = ffmpeg_lavfi(tmp_path / "in.mkv", "-i", str(sample), "-c", "copy")
    ffvw = FFmpegVideoWritter(str(mkv), str(tmp_path / "out.mp4"))  # used to raise KeyError: 'nb_frames'
    # mkv stores millisecond timestamps, so the fps filter may land one frame either way.
    assert abs(copy_all(ffvw) - SAMPLE_SECONDS * 27) <= 1


def test_wrong_frame_size_raises(sample, tmp_path):
    ffvw = FFmpegVideoWritter(str(sample), str(tmp_path / "out.mp4"))
    with ffvw:
        ffvw.write(ffvw.get_frame())
        with pytest.raises(FrameError, match=r"is 100x100 but this video is 320x240"):
            ffvw.write(np.zeros((100, 100, 3), np.uint8))


def test_float_frames_raise_instead_of_turning_black(sample, tmp_path):
    with FFmpegVideoWritter(str(sample), str(tmp_path / "out.mp4")) as ffvw:
        with pytest.raises(FrameError, match=r"0\.\.1"):
            ffvw.write(ffvw.get_frame() / 255.0)
        ffvw.write(ffvw.get_frame())


def test_unusable_output_raises_ffmpegs_reason(sample, tmp_path):
    with pytest.raises(FFmpegError, match=r"output format"):  # used to be a bare BrokenPipeError
        with FFmpegVideoWritter(str(sample), str(tmp_path / "out.notaformat")) as ffvw:
            copy_all(ffvw)


def test_only_rawvideo_pipe_format(sample, tmp_path):
    with pytest.raises(ValueError, match=r"video_format must be 'rawvideo'"):
        FFmpegVideoWritter(str(sample), str(tmp_path / "out.mp4"), video_format="mjpeg")
