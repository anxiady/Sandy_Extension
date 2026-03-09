import os
import tempfile

from faster_whisper import WhisperModel
from flask import Flask, jsonify, request

MODEL_NAME = "small.en"
DEVICE = "cpu"

print("[INIT] Loading Faster Whisper model...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print("Whisper model loaded and ready")

app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            initial_prompt=(
                "Iran Israel Gaza Ukraine Russia NATO politics war technology news"
            ),
        )

        text = ""
        for seg in segments:
            text += seg.text + " "
        return jsonify({"text": text.strip()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    print("[STARTING] Faster Whisper server on port 8804")
    app.run(host="0.0.0.0", port=8804)
