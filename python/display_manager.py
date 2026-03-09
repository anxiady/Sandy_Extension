import queue
import textwrap
import threading

from PIL import Image, ImageDraw, ImageFont

from utils import ImageUtils

_board = None
_worker_started = False
_queue = queue.Queue(maxsize=1)
_state_lock = threading.Lock()


def init_display(board):
    global _board, _worker_started
    with _state_lock:
        _board = board
        if not _worker_started:
            threading.Thread(target=_display_worker, daemon=True).start()
            _worker_started = True


def show_text(message):
    if message is None:
        message = ""
    text = str(message)
    try:
        _queue.put_nowait(text)
    except queue.Full:
        try:
            _queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _queue.put_nowait(text)
        except queue.Full:
            pass


def _display_worker():
    while True:
        message = _queue.get()
        _render_text(message)


def _render_text(message):
    with _state_lock:
        board = _board
    if board is None:
        return

    width = board.LCD_WIDTH
    height = board.LCD_HEIGHT
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    wrapped_lines = textwrap.wrap(message, width=28) or [""]
    text = "\n".join(wrapped_lines[:10])
    draw.multiline_text((12, 12), text, fill=(255, 255, 255), font=font, spacing=6)

    rgb565_data = ImageUtils.image_to_rgb565(image, width, height)
    board.draw_image(0, 0, width, height, rgb565_data)
