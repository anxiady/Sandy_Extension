#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo ".env not found. Copying template..."
  cp .env.template .env
  echo "Please edit .env and add your API keys."
  exit 1
fi

echo "Starting Whisper STT server..."
python3 python/speech-service/whisper-host.py &
WHISPER_PID=$!
trap "kill $WHISPER_PID 2>/dev/null" EXIT

sleep 3
curl -s http://127.0.0.1:8804/ >/dev/null || echo "Whisper server not responding yet"
echo "Whisper server started."

python3 python/chatbot-ui.py
