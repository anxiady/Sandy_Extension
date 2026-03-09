import os
import tempfile
import traceback

from faster_whisper import WhisperModel
from flask import Flask, jsonify, request

MODEL_NAME = "base"
DEVICE = "cpu"

print("[INIT] Loading Faster Whisper model...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")

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
            language=None,
            initial_prompt=(
                "Middle East Iran Israel Gaza Ukraine Russia war news politics"
            ),
        )

        text = ""
        for seg in segments:
            text += seg.text + " "
        return jsonify({"text": text.strip()})
    except Exception as exc:
        print(f"[STT] transcribe error: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    print("[STARTING] Faster Whisper server on port 8804")
    app.run(host="0.0.0.0", port=8804)
