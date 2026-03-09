import queue
import textwrap
import threading

from PIL import Image, ImageDraw, ImageFont

from utils import ImageUtils
from whisplay import WhisplayBoard

_board = None
_worker_started = False
_queue = queue.Queue(maxsize=1)
_state_lock = threading.Lock()
_font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    36,
)


def init_display(board=None):
    global _board, _worker_started
    with _state_lock:
        if board is None and _board is None:
            board = WhisplayBoard()
            board.set_backlight(100)
        _board = board
        if not _worker_started:
            threading.Thread(target=_display_worker, daemon=True).start()
            _worker_started = True


def show_text(message):
    with _state_lock:
        has_board = _board is not None
    if not has_board:
        init_display()
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
        try:
            _render_text(message)
        except Exception as exc:
            print(f"[display] render error: {exc}")


def _render_text(message):
    with _state_lock:
        board = _board
    if board is None:
        return

    width = board.LCD_WIDTH
    height = board.LCD_HEIGHT
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    wrapped_lines = textwrap.wrap(message, width=10) or [""]
    visible_lines = wrapped_lines[:4]

    line_spacing = 12
    line_heights = []
    max_line_width = 0
    for line in visible_lines:
        bbox = draw.textbbox((0, 0), line, font=_font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, line_width)
        line_heights.append(line_height)

    text_block_height = sum(line_heights)
    if len(line_heights) > 1:
        text_block_height += line_spacing * (len(line_heights) - 1)

    start_y = max(0, (height - text_block_height) // 2)

    cursor_y = start_y
    for idx, line in enumerate(visible_lines):
        bbox = draw.textbbox((0, 0), line, font=_font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        line_x = max(0, (width - line_width) // 2)
        draw.text((line_x, cursor_y), line, fill=(255, 255, 255), font=_font)
        cursor_y += line_height
        if idx < len(visible_lines) - 1:
            cursor_y += line_spacing

    rgb565_data = ImageUtils.image_to_rgb565(image, width, height)
    board.draw_image(0, 0, width, height, rgb565_data)
