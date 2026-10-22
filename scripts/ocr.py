"""On-screen text from frames with EasyOCR (GPU when CUDA is present, else CPU).

Reads big text well (titles, captions, slides, diagram labels). Small UI text comes back
as misread noise ("Nevs North Your"), so lines shorter than MIN_HEIGHT px (on a 1024px-wide
frame; measured on a screen recording: noise was <=14px, headlines >=18px) or under MIN_CONF
are dropped. Zoom into the full frame for small text.
"""
from __future__ import annotations

import importlib.util
import re
import sys

MIN_HEIGHT, MIN_CONF, MAX_CHARS, RECENT = 17, 0.4, 8000, 3


def available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("easyocr", "numpy", "PIL"))


def read_frames(paths: list[str]) -> list[list[str]]:
    """Text lines per frame, top to bottom then left to right."""
    import easyocr
    import numpy as np
    import torch
    from PIL import Image

    gpu = torch.cuda.is_available()
    print(f"[watch] reading on-screen text in {len(paths)} frames (EasyOCR, {'GPU' if gpu else 'CPU'})…", file=sys.stderr)
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    out = []
    for path in paths:
        # Pixels, not the path: EasyOCR's cv2 reader returns None for a non-ASCII Windows path.
        with Image.open(path) as img:
            min_height = MIN_HEIGHT * img.width / 1024
            pixels = np.array(img.convert("RGB"))
        boxes = [r for r in reader.readtext(pixels, batch_size=32, min_size=int(min_height))
                 if r[2] >= MIN_CONF and r[0][2][1] - r[0][0][1] >= min_height]
        boxes.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        out.append([text.strip() for _, text, _ in boxes if text.strip()])
    return out


def _key(line: str) -> str:
    return re.sub(r"[^a-z0-9]", "", line.lower())


def report(stamps: list[str], per_frame: list[list[str]]) -> list[str]:
    """Markdown lines for `## On-screen text`: per frame, only lines the last RECENT frames did
    not show, so a held screen prints once even when OCR flickers ("new." / "new:").
    Capped at MAX_CHARS to keep the report readable."""
    out, recent, used = [], [], 0
    for stamp, lines in zip(stamps, per_frame):
        seen = set().union(*recent)
        fresh = [line for line in lines if _key(line) and _key(line) not in seen]
        recent = (recent + [{_key(line) for line in lines}])[-RECENT:]
        if not fresh:
            continue
        line = f"- {stamp} " + " | ".join(fresh)
        if used + len(line) > MAX_CHARS:
            out.append("- … more text cut; re-run with --start/--end for a section.")
            break
        out.append(line)
        used += len(line)
    return out or ["- No readable text."]
