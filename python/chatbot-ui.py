import os
import signal
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from whisplay import WhisplayBoard

load_dotenv()

with open("python/sandy_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

api_key = (os.getenv("KIMI_API_KEY") or "").strip()
base_url = (os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1") or "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
MODEL_NAME = (os.getenv("LLM_MODEL", "moonshot-v1-8k") or "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()

SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
CHANNELS = 1

if not api_key:
    raise RuntimeError("Missing API key. Set KIMI_API_KEY in .env")
if len(api_key) < 24:
    raise RuntimeError(
        f"KIMI_API_KEY looks invalid (length={len(api_key)}). "
        "Check your real key in .env and remove placeholder values."
    )
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is required in .env")

print("[LLM] API key present:", bool(api_key))
print("[LLM] Base URL:", os.getenv("KIMI_BASE_URL"))
print("[LLM] Model:", MODEL_NAME)
print("[LLM] API key length:", len(api_key))

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

_is_recording = False
_recorded_chunks = []
_record_lock = threading.Lock()
_busy_lock = threading.Lock()
_stop_event = threading.Event()


def _audio_callback(indata, frames, time_info, status):
    del frames, time_info
    if status:
        print(f"[audio] {status}")
    with _record_lock:
        if _is_recording:
            _recorded_chunks.append(indata.copy())


def _recording_stream():
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_audio_callback,
    )


def _recording_to_wav_file() -> str:
    with _record_lock:
        if not _recorded_chunks:
            return ""
        audio = np.concatenate(_recorded_chunks, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return temp_path


def transcribe_audio(file_path: str) -> str:
    print("Transcribing...")
    # Local whisper-host exposes POST /recognize and accepts a file path in JSON.
    r = requests.post(
        "http://127.0.0.1:8804/recognize",
        json={"filePath": file_path},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["recognition"]


def ask_llm(transcript: str) -> str:
    print("Thinking...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.7,
    )
    reply = response.choices[0].message.content
    return (reply or "").strip()


def speak_text(text: str) -> None:
    print("Speaking...")
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        f"?output_format=pcm_16000"
    )
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()

    pcm = np.frombuffer(resp.content, dtype=np.int16)
    if pcm.size == 0:
        return

    audio = pcm.astype(np.float32) / 32768.0
    sd.play(audio, samplerate=16000)
    sd.wait()


def process_turn() -> None:
    if not _busy_lock.acquire(blocking=False):
        return

    try:
        audio_file = _recording_to_wav_file()
        if not audio_file:
            print("No audio captured.")
            return

        transcript = transcribe_audio(audio_file)
        if not transcript:
            print("No speech recognized.")
            return

        print(f"You: {transcript}")
        reply = ask_llm(transcript)
        if not reply:
            print("Sandy returned an empty response.")
            return

        print(f"Sandy: {reply}")
        speak_text(reply)
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if "audio_file" in locals() and audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        _busy_lock.release()


def on_button_press() -> None:
    global _is_recording
    with _record_lock:
        _recorded_chunks.clear()
        _is_recording = True
    print("Listening...")


def on_button_release() -> None:
    global _is_recording
    with _record_lock:
        was_recording = _is_recording
        _is_recording = False
    if was_recording:
        threading.Thread(target=process_turn, daemon=True).start()


def shutdown(signum=None, frame=None) -> None:
    del signum, frame
    _stop_event.set()


def main() -> None:
    board = WhisplayBoard()
    board.on_button_press(on_button_press)
    board.on_button_release(on_button_release)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Sandy v1 ready. Hold button to talk, release to send.")

    try:
        with _recording_stream():
            while not _stop_event.is_set():
                time.sleep(0.05)
    finally:
        board.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
