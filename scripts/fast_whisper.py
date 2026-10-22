"""Local transcription with faster-whisper: CUDA GPU when present, else CPU (int8). Audio never leaves the machine.

Needs `faster_whisper` importable by the interpreter running watch.py. The model comes
from the Hugging Face cache (downloaded once if missing).
Sequential decoding, not the batched pipeline: batching merges speech into ~25s
segments, too coarse to line a sentence up with a frame.

Runs in a child process that ends with os._exit: freeing a CUDA WhisperModel after
inference aborts the whole interpreter (ctranslate2 4.7.1 + torch 2.11 on Windows,
"Fatal Python error: Aborted", exit 127), which killed watch.py mid-run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

GPU_MODEL, CPU_MODEL = "large-v3-turbo", "small"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def has_cuda() -> bool:
    try:
        import ctranslate2
    except ImportError:
        return False
    return ctranslate2.get_cuda_device_count() > 0


def transcribe_media(path: str) -> list[dict]:
    name = os.environ.get("WATCH_FW_MODEL") or (GPU_MODEL if has_cuda() else CPU_MODEL)
    print(f"[watch] transcribing on {'GPU' if has_cuda() else 'CPU'} (faster-whisper {name})…", file=sys.stderr)
    out = subprocess.run([sys.executable, "-I", __file__, path, name],
                         capture_output=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0, f"faster-whisper child failed ({out.returncode}): {out.stderr[-2000:]}"
    return json.loads(out.stdout)


if __name__ == "__main__":
    from faster_whisper import WhisperModel

    cuda = has_cuda()
    model = WhisperModel(sys.argv[2], device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    segments, _ = model.transcribe(sys.argv[1], vad_filter=True)
    sys.stdout.write(json.dumps([{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                                 for s in segments if s.text.strip()]))
    sys.stdout.flush()
    os._exit(0)
