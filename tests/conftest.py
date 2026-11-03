"""Shared test helpers.

The tests need a real ffmpeg and ffprobe on PATH (see tests/Dockerfile). Videos are generated
with ffmpeg's built-in test pattern, so the repo ships no sample media.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The sample clip: 2 seconds of 320x240 test pattern at 25 fps with a 440 Hz tone.
SAMPLE_SIZE = (320, 240)
SAMPLE_FPS = 25
SAMPLE_SECONDS = 2


def ffprobe(path) -> dict:
    """Independent check of a file: ffprobe's JSON, not the library's own probe()."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
    ).stdout
    return json.loads(out)


def video_stream(path) -> dict:
    return next(s for s in ffprobe(path)["streams"] if s["codec_type"] == "video")


def make_frames(count: int, width: int, height: int, channels: int = 3):
    """Deterministic frames: a gradient that shifts every frame."""
    ys, xs = np.mgrid[0:height, 0:width]
    for i in range(count):
        base = (xs * 3 + ys * 2 + i * 7) % 256
        planes = [(base + c * 85) % 256 for c in range(channels)]
        yield np.stack(planes, axis=-1).astype(np.uint8)


def ffmpeg_lavfi(path: Path, *args: str) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)
    return path


def rotate_90(src: Path, dst: Path) -> Path:
    """Copy a video and mark it as rotated by 90 degrees, like a phone recording held upright.

    ffmpeg 6.0+ has -display_rotation; older versions (Debian 12 ships 5.1) use the rotate tag.
    """
    try:
        return ffmpeg_lavfi(dst, "-display_rotation", "90", "-i", str(src), "-c", "copy")
    except subprocess.CalledProcessError:
        return ffmpeg_lavfi(dst, "-i", str(src), "-c", "copy", "-metadata:s:v:0", "rotate=90")


@pytest.fixture(scope="session")
def sample(tmp_path_factory) -> Path:
    width, height = SAMPLE_SIZE
    return ffmpeg_lavfi(
        tmp_path_factory.mktemp("media") / "sample.mp4",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={SAMPLE_FPS}:duration={SAMPLE_SECONDS}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={SAMPLE_SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
    )  # fmt: skip


@pytest.fixture
def fake_ffmpeg(tmp_path):
    """Build a stand-in 'ffmpeg' that runs the given Python code, to simulate crashes."""

    def build(code: str) -> str:
        script = tmp_path / "fake_ffmpeg.py"
        script.write_text(textwrap.dedent(code))
        if os.name == "nt":
            wrapper = tmp_path / "ffmpeg.cmd"
            wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
        else:
            wrapper = tmp_path / "ffmpeg"
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        return str(wrapper)

    return build
