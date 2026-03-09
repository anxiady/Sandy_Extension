import queue
import textwrap
import threading
import time

from PIL import Image, ImageDraw, ImageFont

from utils import ImageUtils
from whisplay import WhisplayBoard

_TEXT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_STATE_EMOJI = {
    "idle": "🙂",
    "listening": "👂",
    "thinking": "🤔",
    "speaking1": "😄",
    "speaking2": "😮",
}

_board = None
_worker_started = False
_render_queue = queue.Queue(maxsize=1)
_state_lock = threading.Lock()
_subtitle_font = ImageFont.truetype(_TEXT_FONT_PATH, 16)
_emoji_font = None
_speaking = False
_speaking_thread = None


def _load_emoji_font():
    for path in _EMOJI_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, 88)
        except Exception:
            continue
    return ImageFont.load_default()


_emoji_font = _load_emoji_font()


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
    with _state_lock:
        has_board = _board is not None
    if not has_board:
        init_display()
    _enqueue_render((state or "idle", subtitle or ""))


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
            _render_emoji_ui(state, subtitle)
        except Exception as exc:
            print(f"[display] render error: {exc}")


def _render_emoji_ui(state, subtitle):
    with _state_lock:
        board = _board
    if board is None:
        return

    width = board.LCD_WIDTH
    height = board.LCD_HEIGHT
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    emoji = _STATE_EMOJI.get(state, _STATE_EMOJI["idle"])
    bbox = draw.textbbox((0, 0), emoji, font=_emoji_font)
    emoji_width = bbox[2] - bbox[0]
    emoji_height = bbox[3] - bbox[1]
    emoji_x = max(0, (width - emoji_width) // 2)
    emoji_y = 20
    draw.text((emoji_x, emoji_y), emoji, fill=(255, 255, 255), font=_emoji_font)

    _draw_subtitle(draw, width, height, subtitle)

    rgb565_data = ImageUtils.image_to_rgb565(image, width, height)
    board.draw_image(0, 0, width, height, rgb565_data)


def _draw_subtitle(draw, width, height, subtitle):
    if not subtitle:
        return
    text = str(subtitle).strip()
    if not text:
        return

    lines = textwrap.wrap(text, width=26)[:4]
    panel_top = int(height * 0.55)
    draw.rectangle((0, panel_top, width, height), fill=(0, 0, 0))

    y = panel_top + 8
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=_subtitle_font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = max(0, (width - line_width) // 2)
        draw.text((x, y), line, fill=(255, 255, 255), font=_subtitle_font)
        y += line_height + 4
