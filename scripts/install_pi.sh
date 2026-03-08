#!/bin/bash

set -e

echo "Updating system..."
sudo apt update

echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-dev portaudio19-dev sox ffmpeg git

echo "Installing python dependencies..."
pip3 install -r python/requirements.txt

echo "Setup complete."
