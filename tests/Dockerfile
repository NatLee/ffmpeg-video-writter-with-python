# Test image: Python plus the ffmpeg/ffprobe that Debian ships. Build it from the repo root:
#   docker build -f tests/Dockerfile -t ffmpeg-video-writter-test .
#   docker run --rm ffmpeg-video-writter-test
# Other Python versions: --build-arg PYTHON=3.8 (Debian 12, ffmpeg 5.1) or PYTHON=3.14.
ARG PYTHON=3.12
FROM python:${PYTHON}-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md ffmpeg_video_writer.py FFmpegVideoWritter.py ./
# opencv only runs the draw_rectangles example; the library itself needs numpy alone.
RUN pip install --no-cache-dir ".[test]" opencv-python-headless
COPY tests ./tests
COPY examples ./examples
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
