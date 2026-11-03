"""The original demo: halve every frame of a video, draw a random rectangle on it, save MP4 + GIF.

    python examples/draw_rectangles.py [input.mp4] [--fps 27] [--show]

Without an input it uses bouncing_ball.mp4 from examples/quick_start.py.
Needs OpenCV for resizing and drawing: pip install opencv-python
(opencv-python-headless works too, without --show).
"""

import argparse
from random import randint

import cv2

from ffmpeg_video_writer import VideoReader, VideoWriter, to_gif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", default="bouncing_ball.mp4", help="video to read")
    parser.add_argument("--output", default="result.mp4", help="video to write (default: %(default)s)")
    parser.add_argument("--gif", default="result.gif", help="GIF to write (default: %(default)s)")
    parser.add_argument("--fps", type=float, default=27, help="output frame rate (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="preview the frames in a window")
    args = parser.parse_args()

    with VideoReader(args.input, fps=args.fps) as reader, VideoWriter(args.output, fps=args.fps) as writer:
        for frame in reader:  # RGB frames
            out_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            h, w = out_frame.shape[:2]
            cv2.rectangle(
                out_frame,
                (randint(0, w), randint(0, h)),
                (randint(0, w), randint(0, h)),
                (randint(0, 255), randint(0, 255), randint(0, 255)),
                randint(1, 5),
            )
            if args.show:
                cv2.imshow("frame", cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))  # OpenCV shows BGR
                cv2.waitKey(1)
            writer.write(out_frame)
        print(f"{args.output}: {writer.frames_written} frames, {writer.output_size[0]}x{writer.output_size[1]}")

    # You can get gif output!
    to_gif(args.output, args.gif)
    print(f"{args.gif}: done")


if __name__ == "__main__":
    main()
