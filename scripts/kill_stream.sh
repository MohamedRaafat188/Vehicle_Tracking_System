#!/usr/bin/env bash
# Kill one simulated camera stream started by simulate_cameras.sh, so
# rtsp_reader.py's reconnect/backoff logic can be tested against a genuine
# stream drop instead of a mocked one.
#
# Usage: scripts/kill_stream.sh <n>   (e.g. `kill_stream.sh 2` drops cam2)

set -euo pipefail

N="${1:?usage: kill_stream.sh <camera_number>}"
PID_DIR="$(dirname "$0")/.simulate_cameras_pids"
PID_FILE="$PID_DIR/ffmpeg_cam${N}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "no recorded pid for cam${N} (expected $PID_FILE) - is simulate_cameras.sh running?" >&2
  exit 1
fi

pid="$(cat "$PID_FILE")"
echo "Killing cam${N} (pid $pid)"
kill "$pid" || true
rm -f "$PID_FILE"
echo "cam${N} dropped. rtsp_reader.py should log a disconnect and start retrying with backoff."
