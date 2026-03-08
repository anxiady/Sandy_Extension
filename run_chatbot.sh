#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo ".env not found. Copying template..."
  cp .env.template .env
  echo "Please edit .env and add your API keys."
  exit 1
fi

# Load variables exactly as written in .env (avoids xargs mangling API keys).
set -a
source .env
set +a

python3 python/chatbot-ui.py
