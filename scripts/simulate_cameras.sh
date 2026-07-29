#!/usr/bin/env bash
# Simulate N RTSP cameras on localhost using MediaMTX + ffmpeg, so
# rtsp_reader.py / stream_manager.py can be tested against real RTSP
# connections without physical cameras.
#
# Requires: mediamtx (https://github.com/bluenviron/mediamtx) and ffmpeg on PATH.
# Neither is installed by requirements.txt - they are system tools, not python deps.
#
# Usage: scripts/simulate_cameras.sh <video_file> [num_cameras]

set -euo pipefail

VIDEO_FILE="${1:?usage: simulate_cameras.sh <video_file> [num_cameras]}"
NUM_CAMS="${2:-4}"
RTSP_PORT="${RTSP_PORT:-8554}"
PID_DIR="$(dirname "$0")/.simulate_cameras_pids"

mkdir -p "$PID_DIR"

if ! command -v mediamtx >/dev/null 2>&1; then
  echo "mediamtx not found on PATH. Install it: https://github.com/bluenviron/mediamtx/releases" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH." >&2
  exit 1
fi
if [ ! -f "$VIDEO_FILE" ]; then
  echo "video file not found: $VIDEO_FILE" >&2
  exit 1
fi

echo "Starting mediamtx..."
mediamtx > "$PID_DIR/mediamtx.log" 2>&1 &
echo $! > "$PID_DIR/mediamtx.pid"
sleep 1

for i in $(seq 1 "$NUM_CAMS"); do
  cam="cam${i}"
  echo "Publishing $VIDEO_FILE as rtsp://localhost:${RTSP_PORT}/${cam}"
  # -c copy: no re-encoding, so this doesn't distort CPU-bound benchmark numbers
  ffmpeg -re -stream_loop -1 -i "$VIDEO_FILE" -c copy -f rtsp \
    "rtsp://localhost:${RTSP_PORT}/${cam}" > "$PID_DIR/ffmpeg_${cam}.log" 2>&1 &
  echo $! > "$PID_DIR/ffmpeg_${cam}.pid"
done

echo ""
echo "Streams live at rtsp://localhost:${RTSP_PORT}/cam1 .. cam${NUM_CAMS}"
echo "PIDs recorded in $PID_DIR"
echo "Stop with: kill \$(cat $PID_DIR/*.pid)"
