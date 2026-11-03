"""Run the examples the README shows, in a scratch folder."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import ROOT, video_stream

EXAMPLES = ROOT / "examples"


def run(script, cwd, *args) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_quick_start_then_draw_rectangles(tmp_path):
    out = run("quick_start.py", tmp_path)
    assert "640x360, 90 frames at 30 fps, h264 yuv420p" in out
    assert video_stream(tmp_path / "bouncing_ball.gif")["codec_name"] == "gif"

    pytest.importorskip("cv2")
    out = run("draw_rectangles.py", tmp_path)
    assert "result.mp4: 81 frames, 320x180" in out  # 3 s at 27 fps, half size
    assert video_stream(tmp_path / "result.gif")["width"] == 320


def test_show_errors_prints_every_case(tmp_path):
    out = run("show_errors.py", tmp_path)
    assert out.count("--- ") == 6
    assert out.count("FrameError: ") == 3
    assert out.count("FFmpegError: ") == 2
    assert out.count("FFmpegNotFoundError: ") == 1
