"""Make the classic mistakes on purpose and print the error each one raises.

    python examples/show_errors.py

Piping frames into ffmpeg by hand, each of these ends in a silent quit, a broken pipe or a
broken file. Here every one of them is an exception that says what went wrong.
"""

import os
import tempfile

import numpy as np

from ffmpeg_video_writer import FFmpegError, FrameError, VideoWriter

FRAME = np.zeros((360, 640, 3), np.uint8)


def show(title, action, **writer_options):
    print(f"--- {title}")
    with tempfile.TemporaryDirectory() as folder:
        try:
            with VideoWriter(os.path.join(folder, "demo.mp4"), **writer_options) as video:
                action(video)
        except (FrameError, FFmpegError) as exc:
            print(f"{type(exc).__name__}: {exc}\n")


def different_size(video):
    video.write(FRAME)
    video.write(np.zeros((480, 640, 3), np.uint8))


def float_frames(video):
    video.write(FRAME / 255.0 + 0.5)


def rgba_frame(video):
    video.write(np.zeros((360, 640, 4), np.uint8))


def thirty_frames(video):
    for _ in range(30):
        video.write(FRAME)


def odd_frame(video):
    video.write(np.zeros((361, 641, 3), np.uint8))


show("A frame with a different size halfway through", different_size)
show("Float frames scaled 0..1 (astype(uint8) would make them black)", float_frames)
show("An RGBA frame for an RGB video", rgba_frame)
show("A typo in the codec name", thirty_frames, codec="libx246")
show("An odd size with yuv420p and padding turned off", odd_frame, pad_to_even=False)
show("ffmpeg is not installed", thirty_frames, ffmpeg="ffmpeg-not-installed")
