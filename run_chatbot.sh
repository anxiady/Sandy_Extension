#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo ".env not found. Copying template..."
  cp .env.template .env
  echo "Please edit .env and add your API keys."
  exit 1
fi

echo "Maximizing Whisplay speaker volume..."
CARD_INDEX="$(awk '/wm8960/ {gsub(/[^0-9]/, "", $1); print $1; exit}' /proc/asound/cards 2>/dev/null)"
if [ -n "$CARD_INDEX" ]; then
  amixer -c "$CARD_INDEX" sset Speaker 100% >/dev/null 2>&1 || true
  amixer -c "$CARD_INDEX" sset Speaker unmute >/dev/null 2>&1 || true
else
  for c in 0 1 2; do
    amixer -c "$c" sset Speaker 100% >/dev/null 2>&1 || true
    amixer -c "$c" sset Speaker unmute >/dev/null 2>&1 || true
  done
fi

echo "Starting Whisper STT server..."
python3 python/speech-service/whisper-host.py > whisper.log 2>&1 &
WHISPER_PID=$!
trap "kill $WHISPER_PID 2>/dev/null" EXIT

for i in $(seq 1 60); do
  if ! kill -0 "$WHISPER_PID" 2>/dev/null; then
    echo "Whisper server exited unexpectedly. Last logs:"
    tail -n 50 whisper.log
    exit 1
  fi
  if curl -s -o /dev/null -X POST http://127.0.0.1:8804/recognize \
      -H "Content-Type: application/json" \
      -d '{}'; then
    echo "Whisper server started."
    break
  fi
  sleep 1
done

echo "Starting Sandy..."
python3 python/chatbot-ui.py
