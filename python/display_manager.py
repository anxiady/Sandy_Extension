import os
import queue
import textwrap
import threading
import time

from PIL import Image, ImageDraw, ImageFont

from utils import ImageUtils
from whisplay import WhisplayBoard

AVATAR_DIR = os.path.join(os.path.dirname(__file__), "ui", "avatar")
_SUBTITLE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_board = None
_worker_started = False
_render_queue = queue.Queue(maxsize=1)
_state_lock = threading.Lock()
_subtitle_font = ImageFont.truetype(_SUBTITLE_FONT_PATH, 18)
_current_state = "idle"
_current_subtitle = ""
_speaking = False
_speaking_thread = None


def init_display(board=None):
    global _board, _worker_started
    with _state_lock:
        if board is None and _board is None:
            board = WhisplayBoard()
            board.set_backlight(100)
        _board = board
        if not _worker_started:
            threading.Thread(target=_render_worker, daemon=True).start()
            _worker_started = True


def show_avatar(state, subtitle=""):
    global _current_state, _current_subtitle
    with _state_lock:
        has_board = _board is not None
        _current_state = state or "idle"
        _current_subtitle = subtitle or ""
    if not has_board:
        init_display()
    _enqueue_render((_current_state, _current_subtitle))


def animate_speaking(subtitle=""):
    global _speaking, _speaking_thread
    with _state_lock:
        if _speaking:
            return
        _speaking = True
    _speaking_thread = threading.Thread(
        target=_speaking_loop, args=(subtitle or "",), daemon=True
    )
    _speaking_thread.start()


def stop_speaking_animation():
    global _speaking, _speaking_thread
    with _state_lock:
        _speaking = False
        worker = _speaking_thread
    if worker is not None:
        worker.join(timeout=0.5)
    with _state_lock:
        _speaking_thread = None


def _speaking_loop(subtitle):
    while True:
        with _state_lock:
            if not _speaking:
                break
        show_avatar("speaking1", subtitle)
        time.sleep(0.2)
        with _state_lock:
            if not _speaking:
                break
        show_avatar("speaking2", subtitle)
        time.sleep(0.2)


def _enqueue_render(item):
    try:
        _render_queue.put_nowait(item)
    except queue.Full:
        try:
            _render_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _render_queue.put_nowait(item)
        except queue.Full:
            pass


def _render_worker():
    while True:
        state, subtitle = _render_queue.get()
        try:
            _render_avatar(state, subtitle)
        except Exception as exc:
            print(f"[display] render error: {exc}")


def _render_avatar(state, subtitle):
    with _state_lock:
        board = _board
    if board is None:
        return

    width = board.LCD_WIDTH
    height = board.LCD_HEIGHT
    image = _load_avatar_image(state, width, height)
    draw = ImageDraw.Draw(image)
    _draw_subtitle(draw, width, height, subtitle)

    rgb565_data = ImageUtils.image_to_rgb565(image, width, height)
    board.draw_image(0, 0, width, height, rgb565_data)


def _load_avatar_image(state, width, height):
    path = os.path.join(AVATAR_DIR, f"{state}.png")
    if not os.path.exists(path):
        path = os.path.join(AVATAR_DIR, "idle.png")
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    return image


def _draw_subtitle(draw, width, height, subtitle):
    if not subtitle:
        return
    text = str(subtitle).strip()
    if not text:
        return

    lines = textwrap.wrap(text, width=28)[:2]
    footer_top = height - 46
    draw.rectangle((0, footer_top, width, height), fill=(0, 0, 0))

    y = footer_top + 4
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=_subtitle_font)
        line_width = bbox[2] - bbox[0]
        x = max(0, (width - line_width) // 2)
        draw.text((x, y), line, fill=(255, 255, 255), font=_subtitle_font)
        y += (bbox[3] - bbox[1]) + 2
