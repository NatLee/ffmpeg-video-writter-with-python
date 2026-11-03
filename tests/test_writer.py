"""VideoWriter: the happy path is checked with ffprobe; every failure must raise, never go quiet."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from conftest import ffprobe, make_frames, video_stream

from ffmpeg_video_writer import FFmpegError, FFmpegNotFoundError, FrameError, VideoReader, VideoWriter


def write_all(path, frames, **kwargs):
    with VideoWriter(path, **kwargs) as video:
        for frame in frames:
            video.write(frame)
    return video


# --------------------------------------------------------------------------- happy path


def test_default_output_is_h264_yuv420p_with_exact_frame_count_and_rate(tmp_path):
    out = tmp_path / "out.mp4"
    write_all(out, make_frames(30, 64, 48), fps=15)

    stream = video_stream(out)
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"  # plays in browsers, QuickTime, phones
    assert (stream["width"], stream["height"]) == (64, 48)
    assert int(stream["nb_read_frames"]) == 30
    assert Fraction(stream["avg_frame_rate"]) == 15
    assert float(stream["duration"]) == pytest.approx(2.0, abs=0.01)  # 30 frames / 15 fps


def test_lossless_round_trip_keeps_every_pixel(tmp_path):
    # libx264rgb at crf 0 is lossless RGB, so what comes back must be exactly what went in.
    frames = list(make_frames(5, 32, 24))
    out = tmp_path / "lossless.mkv"
    write_all(out, frames, codec="libx264rgb", crf=0, output_pixel_format="rgb24")

    with VideoReader(out) as reader:
        back = list(reader)
    assert len(back) == len(frames)
    for sent, got in zip(frames, back):
        np.testing.assert_array_equal(sent, got)


def test_bgr_frames_are_stored_with_the_right_colours(tmp_path):
    red_in_bgr = np.zeros((16, 16, 3), np.uint8)
    red_in_bgr[..., 2] = 255  # OpenCV order: blue, green, red
    out = tmp_path / "red.mkv"
    write_all(out, [red_in_bgr] * 3, pixel_format="bgr24", codec="libx264rgb", crf=0, output_pixel_format="rgb24")

    with VideoReader(out) as reader:  # reads RGB
        frame = reader.read()
    assert frame[0, 0].tolist() == [255, 0, 0]


@pytest.mark.parametrize("pixel_format, shape", [("gray", (24, 32)), ("gray", (24, 32, 1)), ("rgba", (24, 32, 4))])
def test_other_pixel_formats(tmp_path, pixel_format, shape):
    out = tmp_path / "out.mp4"
    write_all(out, [np.full(shape, 128, np.uint8)] * 4, pixel_format=pixel_format)
    assert int(video_stream(out)["nb_read_frames"]) == 4


def test_odd_size_is_padded_to_even(tmp_path):
    out = tmp_path / "odd.mp4"
    video = write_all(out, make_frames(3, 33, 21))
    assert video.size == (33, 21)
    assert video.output_size == (34, 22)
    stream = video_stream(out)
    assert (stream["width"], stream["height"]) == (34, 22)
    assert stream["pix_fmt"] == "yuv420p"


def test_crf_and_bitrate_reach_ffmpeg(tmp_path):
    video = write_all(tmp_path / "crf.mp4", make_frames(2, 32, 32), crf=30)
    assert ["-crf", "30"] == video.command[video.command.index("-crf"):][:2]

    video = write_all(tmp_path / "rate.mp4", make_frames(2, 32, 32), crf=30, bitrate="300k")
    assert "-b:v" in video.command and "-crf" not in video.command  # bitrate wins


def test_fractional_fps(tmp_path):
    out = tmp_path / "ntsc.mp4"
    write_all(out, make_frames(30, 32, 32), fps="30000/1001")
    assert Fraction(video_stream(out)["r_frame_rate"]) == Fraction(30000, 1001)


def test_audio_is_muxed_from_another_file(tmp_path, sample):
    out = tmp_path / "with_audio.mp4"
    write_all(out, make_frames(25, 64, 48), fps=25, audio=sample)  # 1 s of video, 2 s of audio

    info = ffprobe(out)
    kinds = sorted(s["codec_type"] for s in info["streams"])
    assert kinds == ["audio", "video"]
    assert float(info["format"]["duration"]) == pytest.approx(1.0, abs=0.15)  # cut to the shorter


def test_lossless_non_uint8_frames_are_accepted(tmp_path):
    # Whole numbers 0..255 in float64/int64 (np.zeros() is float64 by default) convert losslessly.
    frames = [f.astype(dtype) for dtype in (np.float64, np.int64) for f in make_frames(2, 32, 32)]
    video = write_all(tmp_path / "ok.mp4", frames)
    assert video.frames_written == 4


# --------------------------------------------------------------------------- frame errors


def test_wrong_size_mid_stream_raises_and_keeps_the_good_frames(tmp_path):
    out = tmp_path / "out.mp4"
    with VideoWriter(out, fps=10) as video:
        for frame in make_frames(5, 64, 48):
            video.write(frame)
        with pytest.raises(FrameError, match=r"frame 5 is 100x100 but this video is 64x48 \(set by the first frame\)"):
            video.write(np.zeros((100, 100, 3), np.uint8))
        video.write(next(make_frames(1, 64, 48)))  # the writer is still usable
    assert int(video_stream(out)["nb_read_frames"]) == 6


def test_size_argument_is_enforced_from_the_first_frame(tmp_path):
    with VideoWriter(tmp_path / "out.mp4", size=(64, 48)) as video:
        with pytest.raises(FrameError, match=r"frame 0 is 48x64 but this video is 64x48 \(set by size=\)"):
            video.write(np.zeros((64, 48, 3), np.uint8))
        video.write(np.zeros((48, 64, 3), np.uint8))


@pytest.mark.parametrize(
    "frame, message",
    [
        (np.full((8, 8, 3), 0.5), r"looks scaled to 0\.\.1 .*\(frame \* 255\)"),
        (np.full((8, 8, 3), 300, np.int16), r"outside 0\.\.255 .*would wrap around.*np\.clip"),
        (np.full((8, 8, 3), -1.0), r"outside 0\.\.255"),
        (np.full((8, 8, 3), 12.5), r"fractional values.*round\(\)"),
        (np.full((8, 8, 3), np.nan), r"NaN"),
        (np.ones((8, 8, 3), bool), r"dtype bool.*Multiply a mask by 255"),
        (np.full((8, 8, 3), "x"), r"dtype <U1"),
    ],
)
def test_bad_dtype_or_values_raise_with_a_fix(tmp_path, frame, message):
    with VideoWriter(tmp_path / "out.mp4") as video:
        with pytest.raises(FrameError, match=message):
            video.write(frame)
        video.write(np.zeros((8, 8, 3), np.uint8))


@pytest.mark.parametrize(
    "frame, message",
    [
        (np.zeros((8, 8, 4), np.uint8), r"4 channel\(s\) but pixel_format='rgb24' expects 3\. .*'rgba'"),
        (np.zeros((8, 8), np.uint8), r"1 channel\(s\) .*'gray'"),
        (np.zeros((2, 8, 8, 3), np.uint8), r"shape \(2, 8, 8, 3\); expected \(height, width, 3\)"),
        (np.zeros((0, 8, 3), np.uint8), r"empty"),
    ],
)
def test_bad_shape_raises(tmp_path, frame, message):
    with VideoWriter(tmp_path / "out.mp4") as video:
        with pytest.raises(FrameError, match=message):
            video.write(frame)
        video.write(np.zeros((8, 8, 3), np.uint8))


# --------------------------------------------------------------------------- ffmpeg errors


def test_missing_ffmpeg_is_reported_up_front(tmp_path):
    with pytest.raises(FFmpegNotFoundError, match=r"Cannot find 'no-such-ffmpeg'.*apt install ffmpeg"):
        VideoWriter(tmp_path / "out.mp4", ffmpeg="no-such-ffmpeg")


def test_odd_size_without_padding_shows_the_encoder_error(tmp_path):
    with pytest.raises(FFmpegError, match=r"divisible by 2") as caught:
        write_all(tmp_path / "odd.mp4", make_frames(3, 33, 21), pad_to_even=False)
    assert caught.value.returncode not in (0, None)
    assert "libx264" in " ".join(caught.value.cmd)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"output_args": ["-no-such-option", "1"]}, r"no-such-option"),
        ({"codec": "no-such-encoder"}, r"no-such-encoder"),
    ],
)
def test_bad_ffmpeg_option_raises_with_ffmpegs_message(tmp_path, kwargs, message):
    with pytest.raises(FFmpegError, match=message) as caught:
        write_all(tmp_path / "out.mp4", make_frames(20, 64, 48), **kwargs)
    assert "ffmpeg said:" in str(caught.value)


def test_unknown_container_raises(tmp_path):
    with pytest.raises(FFmpegError, match=r"output format"):
        write_all(tmp_path / "out.notaformat", make_frames(3, 64, 48))


def test_missing_output_folder_raises(tmp_path):
    with pytest.raises(FFmpegError, match=r"No such file or directory"):
        write_all(tmp_path / "missing" / "out.mp4", make_frames(3, 64, 48))


def test_ffmpeg_crash_mid_stream_raises_with_its_log(tmp_path, fake_ffmpeg):
    ffmpeg = fake_ffmpeg(
        """
        import sys
        sys.stdin.buffer.read(1000)
        sys.stderr.write("simulated encoder crash: out of unicorns\\n")
        sys.stderr.flush()
        sys.exit(3)
        """
    )
    video = VideoWriter(tmp_path / "out.mp4", ffmpeg=ffmpeg)
    with pytest.raises(FFmpegError, match=r"out of unicorns") as caught:
        for frame in make_frames(50, 320, 240):  # 230 KB per frame: more than the pipe holds
            video.write(frame)
    assert caught.value.returncode == 3
    assert "stopped reading at frame" in str(caught.value) or "exited before frame" in str(caught.value)
    with pytest.raises(ValueError, match="after close"):
        video.write(next(make_frames(1, 320, 240)))


def test_ffmpeg_that_stops_early_is_reported(tmp_path):
    # -frames:v 2 makes ffmpeg quit after two frames with exit code 0. The writer must not
    # pretend the remaining frames were written.
    with pytest.raises(FFmpegError, match=r"ffmpeg (exited before|stopped reading at) frame"):
        write_all(tmp_path / "out.mp4", make_frames(200, 320, 240), output_args=["-frames:v", "2"])


# --------------------------------------------------------------------------- lifecycle


def test_close_without_frames_warns_instead_of_crashing(tmp_path):
    video = VideoWriter(tmp_path / "out.mp4")
    with pytest.warns(RuntimeWarning, match="No frames were written"):
        video.close()
    video.close()  # closing twice is fine


def test_user_exception_wins_and_partial_video_is_kept(tmp_path):
    out = tmp_path / "partial.mp4"
    with pytest.raises(KeyError):
        with VideoWriter(out, fps=10) as video:
            for frame in make_frames(4, 64, 48):
                video.write(frame)
            raise KeyError("boom")
    assert int(video_stream(out)["nb_read_frames"]) == 4


def test_write_after_close(tmp_path):
    video = write_all(tmp_path / "out.mp4", make_frames(1, 8, 8))
    with pytest.raises(ValueError, match="after close"):
        video.write(np.zeros((8, 8, 3), np.uint8))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fps": 0}, "fps must be a positive number"),
        ({"fps": "abc"}, "fps must be a positive number"),
        ({"pixel_format": "yuv420p"}, "pixel_format must be one of"),
        ({"size": (0, 10)}, "size must be two positive numbers"),
    ],
)
def test_bad_arguments_fail_before_ffmpeg_starts(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        VideoWriter(tmp_path / "out.mp4", **kwargs)
