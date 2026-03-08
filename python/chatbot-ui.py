import io
import os
import signal
import sys
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
STT_MODEL = os.getenv("STT_MODEL", "whisper-1").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()

SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
CHANNELS = 1

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in .env")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is required in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

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


def _recording_to_wav_bytes() -> bytes:
    with _record_lock:
        if not _recorded_chunks:
            return b""
        audio = np.concatenate(_recorded_chunks, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def transcribe_audio(wav_data: bytes) -> str:
    print("Transcribing...")
    audio_file = io.BytesIO(wav_data)
    audio_file.name = "recording.wav"
    result = client.audio.transcriptions.create(model=STT_MODEL, file=audio_file)
    return (result.text or "").strip()


def ask_llm(transcript: str) -> str:
    print("Thinking...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return (response.choices[0].message.content or "").strip()


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
        wav_data = _recording_to_wav_bytes()
        if not wav_data:
            print("No audio captured.")
            return

        transcript = transcribe_audio(wav_data)
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
