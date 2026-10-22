"""Tile extracted frames into timestamp-labelled contact sheets.

One sheet holds 9 landscape or 10 portrait frames, so the agent sees a run of
shots side by side in one image read. The full-size frames stay on disk for
close reads of text and code.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from frames import format_time

LANDSCAPE = (3, 3, 512)  # cols, rows, cell width
PORTRAIT = (5, 2, 288)
BLANK_LIT_SHARE = 0.03  # under 3% lit pixels = black card or a lone caption word


def is_blank(path: str) -> bool:
    """True for near-black frames: fades, black gaps, and one-word kinetic caption cards."""
    with Image.open(path) as img:
        small = img.convert("RGB").resize((96, 54))
    lit = sum(1 for px in small.get_flattened_data() if max(px) > 40)
    return lit < BLANK_LIT_SHARE * 96 * 54


def build_sheets(frames: list[dict], out_dir: Path) -> list[dict]:
    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(frames[0]["path"]) as first:
        portrait = first.height > first.width
        ratio = first.height / first.width
    cols, rows, cell_w = PORTRAIT if portrait else LANDSCAPE
    cell_h = round(cell_w * ratio)
    per_sheet = cols * rows
    font = ImageFont.load_default(size=22)
    sheets = []
    for n, i in enumerate(range(0, len(frames), per_sheet), start=1):
        group = frames[i:i + per_sheet]
        used_rows = -(-len(group) // cols)
        sheet = Image.new("RGB", (cols * cell_w, used_rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        for j, frame in enumerate(group):
            x, y = (j % cols) * cell_w, (j // cols) * cell_h
            with Image.open(frame["path"]) as img:
                sheet.paste(img.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS), (x, y))
            label = format_time(frame["timestamp_seconds"])
            if frame.get("reason") == "transcript-cue":
                label += " cue"
            box = draw.textbbox((x + 6, y + 6), label, font=font)
            draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill="black")
            draw.text((x + 6, y + 6), label, fill="yellow", font=font)
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="gray")
        path = out_dir / f"sheet_{n:02d}.jpg"
        sheet.save(path, quality=88)
        sheets.append({"path": str(path), "start": group[0]["timestamp_seconds"],
                       "end": group[-1]["timestamp_seconds"], "count": len(group)})
    return sheets
