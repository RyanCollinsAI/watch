#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import threading
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import ConfigError, frame_cap, get_config, load_gemini_key, resolve_engine  # noqa: E402
import gemini  # noqa: E402
from download import download, fetch_captions, is_url, auth_args  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract_at_timestamps, extract_keyframes, extract_scene_or_uniform, format_time, get_metadata, merge_frames, parse_time, parse_timestamps, validate_controls  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402
from fast_whisper import has_cuda  # noqa: E402
from runtime import configure_stdio  # noqa: E402
from sheets import build_sheets, is_blank  # noqa: E402
import ocr  # noqa: E402

try:
    import audio  # noqa: E402
except ImportError:  # numpy missing: the optional audio timeline is not installed
    audio = None


MEDIA_CACHE = Path.home() / ".cache" / "watch" / "media"
CACHE_DAYS = 14
# Presenter phrases that point at the screen; each match pins a cue frame.
CUE_RE = re.compile(
    r"\b(look at|take a look|you can see|as you can see|right here|over here|on (?:the )?screen|"
    r"let me show|i'll show|check (?:this|it) out|notice|this (?:diagram|chart|graph|slide|table|"
    r"code|file|prompt|page|view|ui|layout|terminal|output))\b", re.I)
MAX_AUTO_CUES = 24
AUDIO_TIMEOUT = 600  # seconds to wait for the audio timeline after frames and OCR are done


def cached_media(info: dict) -> Path | None:
    """Stable per-video media path so zoom re-runs skip the download. Prunes stale entries."""
    if not info.get("id") or not info.get("extractor"):
        return None
    MEDIA_CACHE.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - CACHE_DAYS * 86400
    for old in MEDIA_CACHE.iterdir():
        if old.stat().st_mtime < cutoff:
            old.unlink()
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{info['extractor']}-{info['id']}")
    return MEDIA_CACHE / f"{safe}.mp4"


def auto_cues(segments: list[dict], start: float | None, end: float | None) -> list[float]:
    cues: list[float] = []
    for seg in segments:
        t = (seg["start"] + seg["end"]) / 2
        if (start is not None and t < start) or (end is not None and t > end):
            continue
        if CUE_RE.search(seg["text"]) and (not cues or t - cues[-1] >= 5):
            cues.append(round(t, 2))
    if len(cues) > MAX_AUTO_CUES:
        step = len(cues) / MAX_AUTO_CUES
        cues = [cues[int(i * step)] for i in range(MAX_AUTO_CUES)]
    return cues


MIN_GAP = 6.0


def gap_fill(times: list[float], start: float, end: float, room: int) -> list[float]:
    """Midpoints that split the widest gaps until coverage is under MIN_GAP or room runs out.

    Screen recordings change too little for scene detection, which leaves long blind spans.
    """
    bounds = sorted({start, *times, end})
    out: list[float] = []
    while len(out) < room:
        i = max(range(len(bounds) - 1), key=lambda k: bounds[k + 1] - bounds[k])
        if bounds[i + 1] - bounds[i] < MIN_GAP:
            break
        mid = round((bounds[i] + bounds[i + 1]) / 2, 2)
        bounds.insert(i + 1, mid)
        out.append(mid)
    return sorted(out)


LOCAL_ONLY_FLAGS = (("--detail", "detail"), ("--fps", "fps"), ("--max-frames", "max_frames"),
                    ("--timestamps", "timestamps"), ("--whisper", "whisper"), ("--sub-lang", "sub_lang"))


def run_gemini(args, config, key, start_sec, end_sec, auth) -> int:
    """Google watches the video. No captions, frames, or Whisper; no fallback on failure."""
    ignored = [flag for flag, name in LOCAL_ONLY_FLAGS if getattr(args, name) is not None]
    ignored += [flag for flag, on in (("--no-whisper", args.no_whisper), ("--no-dedup", args.no_dedup)) if on]
    clip = (start_sec, end_sec) if start_sec is not None or end_sec is not None else None
    uploaded, warning, result, error, sent, work = None, None, None, None, "URL sent to Google", None
    try:
        if gemini.is_youtube(args.source):
            video = {"uri": args.source}
        else:
            parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
            if parent:
                parent.mkdir(parents=True, exist_ok=True)
            sent = "not sent to Google"
            work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
            print("[watch] downloading media…" if is_url(args.source) else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", **(auth if is_url(args.source) else {}))
            sent = "upload to Google failed"
            print("[watch] uploading to the Gemini Files API…", file=sys.stderr)
            uploaded = gemini.upload_file(Path(media["video_path"]), key)
            video = {"uri": uploaded["uri"], "mime_type": uploaded["mime_type"]}
            sent = "video uploaded to Google, deleted after the answer"
        print(f"[watch] asking {config['gemini_model']}…", file=sys.stderr)
        result = gemini.ask(video, args.question, model=config["gemini_model"], key=key,
                            clip=clip, timeout=config["gemini_timeout"])
    except SystemExit as exc:
        error = str(exc)
    finally:
        if uploaded:
            warning = gemini.delete_file(uploaded["name"], key)
        if work:  # Run-owned; holds at most a downloaded copy. A local source file lives elsewhere.
            shutil.rmtree(work, ignore_errors=True)

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source} ({sent})")
    print(f"- **Engine:** {config['gemini_model']} ({result['processing'] if result else 'failed'})")
    if clip:
        print(f"- **Focus range:** {format_time(start_sec or 0)} → {format_time(end_sec) if end_sec is not None else 'end'}")
    if ignored:
        print(f"- **Ignored local options:** {', '.join(ignored)} (these only apply with --engine local)")
    if result and result["total_tokens"] is not None:
        print(f"- **Gemini tokens:** {result['total_tokens']}")
    if warning:
        print(f"- **Cleanup warning:** {warning}")
    print()
    if error:
        print("## Unavailable evidence")
        print()
        print(f"- {error}")
        return 1
    print("## Answer (from Gemini)")
    print()
    print("_These are Gemini's observations of the video, not frames you viewed yourself. "
          "Relay them as such; rerun with `--engine local` to inspect frames directly._")
    print()
    print(result["text"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=None, help="Override frame cap")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="Full-size frame width in pixels (default 1024; contact sheets downscale)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument(
        "--detail",
        choices=["transcript", "efficient", "balanced", "token-burner"],
        default=None,
        help="Fidelity/speed dial: transcript (no frames), efficient (fast keyframes, cap 50), "
             "balanced (scene, cap 100), token-burner (scene, uncapped).",
    )
    ap.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Comma-separated absolute timestamps (SS, MM:SS, HH:MM:SS) to grab a frame at, "
             "e.g. transcript-flagged 'look here' moments. Added on top of the detail frames "
             "(reserved against the cap); with --detail transcript these become the only frames.",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable all local/cloud transcription fallbacks; native captions still work.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai", "whisperx", "fasterwhisper"],
        default=None,
        help="Select the fallback backend for this run; native captions still come first.",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate frame removal. Keeps visually identical "
             "frames (static screen recordings, held slides) instead of collapsing them.",
    )
    ap.add_argument("--sub-lang", default=None,
                    help="Exact caption language preference (default auto/native). With the fasterwhisper "
                         "backend captions are skipped unless this is set.")
    ap.add_argument("--no-audio", action="store_true", help="Skip the music, laughter and sound-effect timeline")
    ap.add_argument("--no-ocr", action="store_true", help="Skip reading on-screen text from frames")
    cookies = ap.add_mutually_exclusive_group()
    cookies.add_argument("--cookies", default=None, help="Explicit cookie file (yt-dlp may update this jar)")
    cookies.add_argument("--cookies-from-browser", default=None, help="Explicit yt-dlp browser selector")
    ap.add_argument("--engine", choices=["auto", "gemini", "local"], default=None,
                    help="gemini: Google watches the video and answers (needs GEMINI_API_KEY). "
                         "local: frames + transcript on this machine. Default auto: gemini when a key exists.")
    ap.add_argument("--question", default=None,
                    help="The user's question. Sent to Gemini on a gemini run; unused by the local engine.")
    args = ap.parse_args()
    if args.no_whisper and args.whisper:
        ap.error("--no-whisper conflicts with --whisper")

    config = get_config(backend_override="none" if args.no_whisper else args.whisper)
    detail = args.detail or config["detail"]
    max_frames = args.max_frames if args.max_frames is not None else frame_cap(detail)
    budget_cap = max_frames if max_frames is not None else 100
    start_sec, end_sec = parse_time(args.start), parse_time(args.end)
    validate_controls(args.resolution, max_frames, start_sec, end_sec, args.fps)
    cue_timestamps = parse_timestamps(args.timestamps)
    backend_choice = "none" if args.no_whisper else args.whisper or config["whisper_backend"]
    cookies_file = args.cookies if args.cookies is not None else (None if args.cookies_from_browser else config["cookies_file"])
    cookies_browser = args.cookies_from_browser if args.cookies_from_browser is not None else (None if args.cookies else config["cookies_from_browser"])
    auth_args(cookies_file, cookies_browser)  # Validate even before starting network work.
    auth = {"cookies_file": cookies_file, "cookies_from_browser": cookies_browser}

    gemini_key = load_gemini_key()
    if resolve_engine(args.engine or config["engine"], bool(gemini_key)) == "gemini":
        return run_gemini(args, config, gemini_key, start_sec, end_sec, auth)

    parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
    print(f"[watch] working dir: {work}", file=sys.stderr)
    url_source = is_url(args.source)
    dl = {"subtitle_path": None, "info": {}, "downloaded": False}
    errors = []
    all_segments = []
    track_available = False
    transcript_source = None
    video_path = None
    visual_error = None
    gaps = []

    # GPU Whisper beats YouTube's auto captions, and YouTube answers caption requests here with HTTP 429.
    skip_captions = url_source and backend_choice == "fasterwhisper" and not args.sub_lang and has_cuda()
    if url_source:
        print("[watch] checking metadata via yt-dlp (captions skipped: GPU Whisper)…" if skip_captions
              else "[watch] checking metadata/captions via yt-dlp…", file=sys.stderr)
        dl = fetch_captions(args.source, work / "download", sub_lang=args.sub_lang or config["sub_lang"],
                            captions=not skip_captions, **auth)
        errors.extend(dl.get("errors", []))
        if dl.get("subtitle_path"):
            try:
                all_segments = parse_vtt(dl["subtitle_path"])
                track_available = bool(all_segments)
                track = dl.get("caption_track") or {}
                transcript_source = (f"captions ({track.get('language', 'unknown')}, "
                                     f"{track.get('kind', 'unknown')}, {track.get('provenance', 'unknown')})")
            except (OSError, ValueError) as exc:
                errors.append(f"Caption parsing failed: {exc}")

    audio_only = detail == "transcript" and not cue_timestamps
    # Captions are assessed before range filtering: a quiet interval is not a missing track.
    need_media = not (audio_only and (track_available or backend_choice == "none") and url_source)
    cache_path = cached_media(dl.get("info") or {}) if url_source and not audio_only else None
    if need_media and cache_path and cache_path.is_file():
        os.utime(cache_path)
        video_path = dl["video_path"] = str(cache_path)
        print(f"[watch] using cached media: {cache_path}", file=sys.stderr)
    elif need_media:
        try:
            print("[watch] downloading media…" if url_source else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", audio_only=audio_only,
                             **({"context": dl, **auth} if url_source else {}))
            dl.update(media)
            if cache_path:
                shutil.move(dl["video_path"], cache_path)
                dl["video_path"] = str(cache_path)
            video_path = dl["video_path"]
        except SystemExit as exc:
            visual_error = f"Media unavailable: {exc}"
            errors.append(visual_error)

    try:
        duration = float((dl.get("info") or {}).get("duration") or 0)
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
    except (TypeError, ValueError):
        duration = 0.0
    meta = {"duration_seconds": duration, "width": None, "height": None, "codec": None,
            "has_audio": False, "has_video": False}
    if video_path:
        try:
            meta = get_metadata(video_path)
        except SystemExit as exc:
            visual_error = f"Visuals/audio metadata unavailable: {exc}"
            errors.append(visual_error)
    full_duration = meta["duration_seconds"]
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")
    effective_start = start_sec or 0.0
    effective_end = min(end_sec, full_duration) if end_sec is not None and full_duration > 0 else end_sec or full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=budget_cap)
    else:
        fps, target = auto_fps(effective_duration, max_frames=budget_cap)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    # The audio timeline runs on its own thread (GPU) while Whisper, frames and OCR run here.
    # Daemon, so an early exit of main() never waits on Demucs or Shazam. Transcript-only runs skip it.
    audio_job, audio_box = None, {}
    want_audio = not args.no_audio and not audio_only and video_path and meta.get("has_audio")
    if want_audio and not (audio and audio.available()):
        print("[watch] audio timeline skipped: needs an NVIDIA GPU and the extras in requirements-extras.txt", file=sys.stderr)
        want_audio = False
    if want_audio:
        saved_audio = cache_path.with_suffix(".audio.json") if cache_path else None
        has_video = bool(meta.get("has_video", meta.get("width")))

        def run_audio():
            try:
                audio_box["result"] = audio.cached(video_path, saved_audio, work / "audio", has_video)
            except Exception as exc:  # noqa: BLE001 - optional evidence; reported, never silent
                audio_box["error"] = f"{type(exc).__name__}: {exc}"

        audio_job = threading.Thread(target=run_audio, daemon=True)
        audio_job.start()

    # Transcribe before frame selection so spoken screen cues can pin frames.
    transcript_state = "missing"
    if not track_available and backend_choice != "none" and video_path and meta.get("has_audio"):
        if backend_choice in ("whisperx", "fasterwhisper"):
            backend, api_key = backend_choice, None
        else:
            backend, api_key = load_api_key(None if backend_choice == "auto" else backend_choice)
        # Only the local GPU transcript is cached: it is deterministic and free to keep beside the media.
        saved = cache_path.with_suffix(".transcript.json") if cache_path and backend == "fasterwhisper" else None
        if saved and saved.is_file():
            all_segments, used_backend = json.loads(saved.read_text(encoding="utf-8")), "fasterwhisper"
            track_available, transcript_state = True, "available" if all_segments else "no speech"
            transcript_source = "whisper (fasterwhisper, cached)"
        elif backend:
            try:
                all_segments, used_backend = transcribe_video(video_path, work / "audio.mp3", backend=backend, api_key=api_key)
                if saved:
                    saved.write_text(json.dumps(all_segments), encoding="utf-8")
                gaps = getattr(all_segments, "gaps", [])
                track_available = True
                transcript_state = "no speech" if not all_segments else "available"
                if used_backend == "whisperx":
                    language = config["whisperx_language"] or "auto (unverified)"
                    transcript_source = f"whisper (whisperx {config['whisperx_model']}, language {language})"
                else:
                    transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                transcript_state = "failed"
                errors.append(f"Transcription failed: {exc}")
        else:
            transcript_state = f"unavailable: no matching API key for {backend_choice}"
    elif not track_available and backend_choice == "none":
        transcript_state = "fallback disabled; no captions available"
    elif not track_available and video_path and not meta.get("has_audio"):
        transcript_state = "no audio stream" if not visual_error else "audio metadata unavailable"

    spoken_cues = auto_cues(all_segments, start_sec, end_sec) if detail != "transcript" and video_path else []
    if spoken_cues:
        cue_timestamps = sorted(set(cue_timestamps or []) | set(spoken_cues))

    frames, cue_frames = [], []
    frame_meta = {"engine": "none", "candidate_count": 0, "selected_count": 0, "fallback": False}
    cue_meta = {}
    detail_budget = max_frames
    if video_path and meta.get("has_video", bool(meta.get("width"))):
        try:
            if cue_timestamps:
                cue_frames, cue_meta = extract_at_timestamps(
                    video_path, work / "frames", cue_timestamps, resolution=args.resolution,
                    max_frames=max_frames, start_seconds=start_sec, end_seconds=effective_end or end_sec)
            detail_budget = None if max_frames is None else max_frames - len(cue_frames)
            if detail != "transcript" and detail_budget != 0:
                kwargs = dict(resolution=args.resolution, max_frames=detail_budget,
                              start_seconds=start_sec, end_seconds=effective_end or end_sec, dedup=not args.no_dedup)
                if detail == "efficient":
                    frames, frame_meta = extract_keyframes(video_path, work / "frames", **kwargs)
                else:
                    frames, frame_meta = extract_scene_or_uniform(video_path, work / "frames", fps=fps, target_frames=target, **kwargs)
        except SystemExit as exc:
            visual_error = f"Visual extraction unavailable: {exc}"
            errors.append(visual_error)
    elif video_path and not visual_error and (detail != "transcript" or cue_timestamps):
        visual_error = "No video stream; visual evidence is unavailable."
    if cue_frames:
        frames = merge_frames(frames, cue_frames)
    blank_times = [f["timestamp_seconds"] for f in frames if is_blank(f["path"])]
    frames = [f for f in frames if f["timestamp_seconds"] not in blank_times]
    fill_frames = []
    room = 0 if max_frames is None or detail == "transcript" else max_frames - len(frames)
    if frames and room > 0:
        fill_times = gap_fill([f["timestamp_seconds"] for f in frames], effective_start,
                              effective_end or full_duration, room)
        if fill_times:
            fill_frames, _ = extract_at_timestamps(
                video_path, work / "fill", fill_times, resolution=args.resolution,
                start_seconds=start_sec, end_seconds=effective_end or end_sec)
            for f in fill_frames:
                f["reason"] = "gap-fill"
            blank_fill = [f["timestamp_seconds"] for f in fill_frames if is_blank(f["path"])]
            blank_times += blank_fill
            fill_frames = [f for f in fill_frames if f["timestamp_seconds"] not in blank_fill]
            frames = merge_frames(frames, fill_frames)
    sheets = build_sheets(frames, work / "sheets")
    screen_text = None
    run_ocr = frames and not args.no_ocr
    if run_ocr and not ocr.available():
        print("[watch] on-screen text skipped: pip install easyocr (see requirements-extras.txt)", file=sys.stderr)
        run_ocr = False
    if run_ocr:
        try:
            screen_text = ocr.read_frames([f["path"] for f in frames])
        except Exception as exc:  # noqa: BLE001 - optional evidence; reported, never silent
            errors.append(f"On-screen text unavailable: {type(exc).__name__}: {exc}")
    audio_result = None
    if audio_job:
        audio_job.join(AUDIO_TIMEOUT)
        audio_result = audio_box.get("result")
        if audio_job.is_alive():
            errors.append(f"Audio timeline unavailable: still running after {AUDIO_TIMEOUT}s")
        elif "error" in audio_box:
            errors.append(f"Audio timeline unavailable: {audio_box['error']}")

    transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
    transcript_text = format_transcript(transcript_segments)
    if track_available:
        transcript_state = "available" if transcript_segments else ("no speech in selected range" if focused and all_segments else "no speech")
    for error in errors:
        print(f"[watch] {error}", file=sys.stderr)

    info = dl.get("info") or {}

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if video_path:
        print(f"- **Local media:** `{video_path}`")
    if visual_error:
        print(f"- **Visual status:** {visual_error}")
    if errors:
        print("- **Result:** partial evidence" if frames or transcript_segments else "- **Result:** unavailable evidence")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    range_mode = "focused" if focused else "full"
    print(f"- **Detail:** {detail}")
    detail_count = frame_meta.get("selected_count", 0)
    if detail != "transcript":
        cap_label = "unlimited" if detail_budget is None else str(detail_budget)
        engine = frame_meta.get("engine", "scene")
        fallback = " fallback" if frame_meta.get("fallback") else ""
        deduped = frame_meta.get("deduped_count", 0)
        dedup_note = f", {deduped} near-duplicate{'s' if deduped != 1 else ''} dropped" if deduped else ""
        print(
            f"- **Frames:** {detail_count} selected from {frame_meta.get('candidate_count', detail_count)} "
            f"candidates ({engine}{fallback}{dedup_note}, {range_mode} range, budget {target}, cap {cap_label})"
        )
    elif not cue_frames:
        print("- **Frames:** skipped (transcript detail)")
    if cue_frames:
        dropped = cue_meta.get("dropped_out_of_window", 0)
        drop_note = f", {dropped} dropped outside range" if dropped else ""
        spoken_note = f", {len(spoken_cues)} auto-pinned from spoken screen cues" if spoken_cues else ""
        print(
            f"- **Cue frames:** {len(cue_frames)} at transcript-flagged timestamps "
            f"(transcript-cue{drop_note}{spoken_note})"
        )
    if blank_times:
        print(f"- **Near-black frames dropped:** {len(blank_times)} (black cards, fades, lone caption words) at "
              + ", ".join(format_time(t) for t in sorted(blank_times)))
    if fill_frames:
        print(f"- **Gap-fill frames:** {len(fill_frames)} at the midpoints of the widest gaps between selected frames")
    if frame_meta.get("engine") == "scene" and effective_duration > 0:
        cuts = frame_meta["candidate_count"]
        print(f"- **Pacing:** {cuts} scene changes (~{cuts / effective_duration * 60:.1f}/min, "
              f"avg shot {effective_duration / cuts:.1f}s)")
    if frames:
        print(f"- **Frame size:** max {args.resolution}px wide, max 1998px tall")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print(f"- **Transcript:** {transcript_state}")
    if gaps:
        print("- **Transcript status:** partial; missing intervals:")
        for gap in gaps:
            end = gap["end"] if gap["end"] is not None else full_duration
            print(f"  - {format_time(gap['start'])} → {format_time(end)}")
    if skip_captions:
        print("- **Captions:** skipped; GPU Whisper is configured (pass `--sub-lang en` to use captions)")
    audio_range = (effective_start, effective_end or full_duration)
    if audio_result:
        print(f"- **Audio:** {audio.summary(audio_result, *audio_range)}")

    if detail == "token-burner" and len(frames) > 250:
        print()
        print(
            f"> **Warning:** token-burner detail selected {len(frames)} frames. "
            "This may use a large number of image tokens."
        )

    if not focused and full_duration > 600 and detail not in ("transcript", "token-burner"):
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length "
            f"under `{detail}` detail — its cap spreads thin across the full clip. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a section, or use "
            "`--detail token-burner` to keep every scene-change frame across the whole video."
        )

    chapters = info.get("chapters") or []
    if chapters:
        print()
        print("## Chapters")
        print()
        for ch in chapters:
            print(f"- {format_time(ch.get('start_time') or 0)} {ch.get('title', '')}")
    description = (info.get("description") or "").strip()
    if description:
        print()
        print("## Description")
        print()
        print(description[:1500] + (" …" if len(description) > 1500 else ""))

    if audio_result:
        print()
        print("## Audio")
        print()
        print("\n".join(audio.report(audio_result, *audio_range)))
    if screen_text is not None:
        print()
        print("## On-screen text")
        print()
        print("_EasyOCR on the full frames: big text only, and only lines that are new since the previous frame. "
              "OCR misreads words; for small text, read the full frame._")
        print()
        print("\n".join(ocr.report([format_time(f["timestamp_seconds"]) for f in frames], screen_text)))

    print()
    print("## Contact sheets")
    print()
    if sheets:
        print(
            "**Read every sheet below.** Each tiles frames in time order, left to right, top to bottom, "
            "with the source timestamp in yellow (`cue` = pinned from a spoken screen cue)."
        )
        print()
        for sheet in sheets:
            print(f"- `{sheet['path']}` ({format_time(sheet['start'])} → {format_time(sheet['end'])}, "
                  f"{sheet['count']} frames)")
    else:
        print("_No sheets._")

    print()
    print("## Frames")
    print()
    if frames:
        print(f"Frames live at: `{work / 'frames'}`")
        print()
        print(
            f"Full-size frames ({args.resolution}px wide). **Read one only for a close look** at text, code "
            "or detail a sheet cell is too small for. `t=MM:SS` is the absolute timestamp in the source video."
        )
        print()
        for frame in frames:
            print(
                f"- `{frame['path']}` "
                f"(t={format_time(frame['timestamp_seconds'])}, reason={frame.get('reason', 'selected')})"
            )
    else:
        print("_No frames extracted._")

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        print()
        print("```")
        print(transcript_text)
        print("```")
    else:
        print(f"_{transcript_state.capitalize()}._")
        if detail == "transcript" and not track_available:
            print("_Re-run with `--detail balanced` for visual evidence, or configure a transcription backend with setup.py._")
    if errors:
        print()
        print("## Unavailable evidence")
        for error in errors:
            print(f"- {error}")

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")

    return 1 if errors and not frames and not transcript_segments and not track_available else 0


if __name__ == "__main__":
    configure_stdio()
    try:
        raise SystemExit(main())
    except (ConfigError, OSError) as exc:
        raise SystemExit(str(exc)) from None
