"""Quick start: draw a bouncing ball with numpy, save it as MP4, then make a small GIF.

    python examples/quick_start.py

Writes bouncing_ball.mp4 and bouncing_ball.gif to the current folder.
"""

import numpy as np

from ffmpeg_video_writer import VideoWriter, probe, to_gif

WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 30, 3
RADIUS = 40


def frames():
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    background = np.empty((HEIGHT, WIDTH, 3), np.uint8)  # a dark blue-violet gradient
    background[..., 0] = 30 + 50 * xs / WIDTH
    background[..., 1] = 25 + 35 * ys / HEIGHT
    background[..., 2] = 70 + 90 * xs / WIDTH
    for i in range(FPS * SECONDS):
        t = i / (FPS * SECONDS)
        # The ball bounces off the walls: a triangle wave in x and y.
        x = RADIUS + abs((2 * t * 3) % 2 - 1) * (WIDTH - 2 * RADIUS)
        y = RADIUS + abs((2 * t * 2) % 2 - 1) * (HEIGHT - 2 * RADIUS)
        frame = background.copy()
        frame[(xs - x) ** 2 + (ys - y) ** 2 < RADIUS**2] = (255, 196, 0)
        yield frame


with VideoWriter("bouncing_ball.mp4", fps=FPS) as video:  # H.264 + yuv420p: plays everywhere
    for frame in frames():
        video.write(frame)

info = probe("bouncing_ball.mp4")
print(
    f"bouncing_ball.mp4: {info.width}x{info.height}, {info.frames} frames at {info.fps:g} fps,"
    f" {info.codec} {info.pixel_format}"
)
to_gif("bouncing_ball.mp4", "bouncing_ball.gif", fps=15, width=320)
print("bouncing_ball.gif: done")
