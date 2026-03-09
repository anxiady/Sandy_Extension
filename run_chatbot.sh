#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo ".env not found. Copying template..."
  cp .env.template .env
  echo "Please edit .env and add your API keys."
  exit 1
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
