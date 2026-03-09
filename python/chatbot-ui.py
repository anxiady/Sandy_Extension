import os
import pygame
import signal
import tempfile
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from display_manager import (
    animate_speaking,
    init_display,
    show_avatar,
    stop_speaking_animation,
)
from audio_clean import clean_audio as clean_audio_file
from whisplay import WhisplayBoard

load_dotenv()
sd.default.blocksize = 256

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
TURN_RECORD_SECONDS = float(os.getenv("TURN_RECORD_SECONDS", "4.0"))

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
conversation_active = False
mic_enabled = True
_recorded_chunks = []
conversation_history = []
_record_lock = threading.Lock()
_state_lock = threading.Lock()
_busy_lock = threading.Lock()
_stop_event = threading.Event()
_board = None
whisplay = None
_is_speaking = False


def _audio_callback(indata, frames, time_info, status):
    del frames, time_info
    try:
        # Ignore overflow/underflow warnings to keep capture loop stable.
        if status and not status.input_overflow:
            print(f"[audio] {status}")
        with _record_lock:
            if _is_recording and mic_enabled:
                _recorded_chunks.append(indata.copy())
    except Exception as exc:
        print(f"[audio] callback error: {exc}")


def _recording_stream():
    try:
        return sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=256,
            latency="low",
            callback=_audio_callback,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to open microphone stream: {exc}") from exc


def _recording_to_wav_file() -> str:
    with _record_lock:
        if not _recorded_chunks:
            return ""
        audio = np.concatenate(_recorded_chunks, axis=0)
    pcm16 = audio.astype(np.int16)

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return temp_path


def transcribe_audio(file_path: str) -> str:
    show_avatar("thinking", "Transcribing...")
    print("Transcribing...")
    with open(file_path, "rb") as f:
        r = requests.post(
            "http://127.0.0.1:8804/transcribe",
            files={"audio": f},
            timeout=90,
        )
    if not r.ok:
        raise RuntimeError(f"STT server error {r.status_code}: {r.text}")
    return r.json()["text"]


def ask_llm(transcript: str) -> str:
    show_avatar("thinking", transcript[:80])
    print("Thinking...")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history[-6:]
    messages.append(
        {
            "role": "user",
            "content": f"Answer briefly in one or two sentences.\n{transcript}",
        }
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=120,
    )
    reply = response.choices[0].message.content
    return (reply or "").strip()


def normalize_query(text: str) -> str:
    show_avatar("thinking", "Interpreting...")
    prompt = f"""
Correct speech recognition mistakes but keep the meaning.

Examples:
iron war -> Iran war
apple vision bro -> Apple Vision Pro
tesla mask -> Tesla Musk

Transcript:
{text}

Corrected query:
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You fix speech recognition errors."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    corrected = response.choices[0].message.content
    return (corrected or "").strip()


def needs_clarification(query: str) -> bool:
    ambiguous_terms = [
        "war",
        "news",
        "conflict",
        "updates",
        "latest",
        "politics",
    ]
    words = query.lower().split()

    if len(words) <= 3:
        for t in ambiguous_terms:
            if t in query.lower():
                return True

    return False


def speak_text(text: str) -> None:
    global _is_speaking
    print("Speaking...")
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        f"?output_format=mp3_44100_128"
    )
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    if not resp.content:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(resp.content)
        audio_file = f.name

    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    _is_speaking = True
    try:
        while pygame.mixer.music.get_busy():
            if whisplay is not None and whisplay.button_pressed():
                pygame.mixer.music.stop()
                print("Speech interrupted")
                break
            time.sleep(0.05)
    finally:
        _is_speaking = False
        try:
            os.remove(audio_file)
        except Exception:
            pass


def speak(text: str) -> None:
    speak_text(text)


def is_conversation_active() -> bool:
    with _state_lock:
        return conversation_active


def record_audio() -> str:
    global _is_recording
    if not mic_enabled:
        return ""
    show_avatar("listening")
    print("Listening...")
    with _record_lock:
        _recorded_chunks.clear()
        _is_recording = True
    end_time = time.time() + TURN_RECORD_SECONDS
    while time.time() < end_time and not _stop_event.is_set():
        if not is_conversation_active():
            break
        time.sleep(0.05)
    with _record_lock:
        _is_recording = False
    if not is_conversation_active():
        return ""
    return _recording_to_wav_file()


def process_turn() -> None:
    global mic_enabled
    if not _busy_lock.acquire(blocking=False):
        return

    audio_file = ""
    try:
        audio_file = record_audio()
        if not audio_file:
            print("No audio captured.")
            return

        clean_audio_path = f"{audio_file}.clean.wav"
        clean_audio_file(audio_file, clean_audio_path)
        raw_transcript = transcribe_audio(clean_audio_path)
        if not raw_transcript:
            show_avatar("idle", "No speech recognized.")
            print("No speech recognized.")
            return

        normalized_query = normalize_query(raw_transcript)
        if not normalized_query:
            normalized_query = raw_transcript

        print("Raw:", raw_transcript)
        print("Normalized:", normalized_query)
        transcript = normalized_query

        if needs_clarification(transcript):
            show_avatar("speaking1", "Do you mean the most recent events?")
            mic_enabled = False
            try:
                animate_speaking("Do you mean the most recent events?")
                speak("Do you mean the most recent events?")
            finally:
                stop_speaking_animation()
                mic_enabled = True
            return

        reply = ask_llm(transcript)
        if not reply:
            show_avatar("idle", "No response.")
            print("Sandy returned an empty response.")
            return

        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Sandy: {reply}")
        show_avatar("speaking1", reply[:120])
        mic_enabled = False
        try:
            animate_speaking(reply[:120])
            speak_text(reply)
        finally:
            stop_speaking_animation()
            mic_enabled = True
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        clean_audio_path = f"{audio_file}.clean.wav"
        if audio_file and os.path.exists(clean_audio_path):
            os.remove(clean_audio_path)
        _busy_lock.release()


def on_button_press() -> None:
    global conversation_active
    if _is_speaking:
        return
    with _state_lock:
        conversation_active = not conversation_active
        active = conversation_active
    if active:
        show_avatar("idle", "Conversation mode ON")
        print("Conversation mode ON")
    else:
        show_avatar("idle")
        print("Conversation mode OFF")


def shutdown(signum=None, frame=None) -> None:
    del signum, frame
    stop_speaking_animation()
    _stop_event.set()


def main() -> None:
    global _board, whisplay
    board = WhisplayBoard()
    board.set_backlight(100)
    _board = board
    whisplay = board
    board.on_button_press(on_button_press)
    init_display(board)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    show_avatar("idle", "Sandy Ready")
    print("Sandy v1 ready. Press button to toggle conversation mode.")

    try:
        with _recording_stream():
            while not _stop_event.is_set():
                if is_conversation_active():
                    process_turn()
                else:
                    time.sleep(0.05)
    finally:
        board.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
