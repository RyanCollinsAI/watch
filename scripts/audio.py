"""Audio timeline: music stretches (genre/mood, BPM, Shazam song, level under the voice),
laughter, and sound effects. Runs on the local GPU; only 12s clips go to Shazam.

Demucs splits the audio into voice and everything else first. The AudioSet tagger (AST,
527 classes) then scores 1s windows of each part: the voice part for laughter and vocal
sounds, the rest for music and effects, so a voiceover no longer hides a quiet music bed.
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

VERSION = 7  # bump when analyze() output changes, so cached .audio.json files recompute
TAGGER = "MIT/ast-finetuned-audioset-10-10-0.4593"
EVENT_LINES = 80
TAG_SR, HQ_SR = 16000, 44100
WIN = 1.0
SPEECH = range(0, 8)
LAUGHTER = range(16, 22)
VOCAL_EVENT = [*range(8, 27), *range(38, 51)]  # shouts, screams, laughs, gasps, coughs
SINGING = range(27, 38)
MUSIC = 137
MUSIC_DETAIL = [*SINGING, *range(138, 283)]
AMBIENT = {500, 501, 502, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
           520, 521, 522, 523, 524, 525, 526}
NOT_EFFECT = {*SPEECH, *VOCAL_EVENT, *SINGING, *range(MUSIC, 283), *AMBIENT}
MUSIC_ON, MUSIC_SMOOTH, MUSIC_MIN, MUSIC_JOIN = 0.2, 5, 3.0, 2.0
EVENT_ON, EVENT_DB = 0.4, 15.0
# The tagger scores real laughter low (0.3-0.6, measured on a Friends laugh track and a laughing
# compilation), and a laugh track sits ~20 dB under the dialogue, so laughter gets a looser bar.
LAUGH_ON, LAUGH_DB = 0.3, 25.0
HIT_FRAME, HIT_MIN_DB, HIT_RISE_DB, HIT_HIGH_RISE_DB, HIT_MUSIC_RISE_DB = 0.05, -25.0, 15.0, 10.0, 20.0
HIT_OFF_CUT_MIN_DB = -18.0
HIT_NAME_ON = 0.05  # a quiet hit's tagger label below this prints as "unnamed"
CUT_SCENE, CUT_NEAR = 0.3, 0.25  # ffmpeg scene score for a cut; a hit within this many seconds is "on a cut"
# Shazam rejects calls (non-JSON body) after ~20 in a row, so one video spends at most
# SHAZAM_PROBES + 2 * SHAZAM_RETRIES = 20 calls.
SHAZAM_CLIP, SHAZAM_EVERY, SHAZAM_PROBES, SHAZAM_RETRIES, CONFIRM_SHIFT = 12.0, 10.0, 14, 3, 5.0
SHAZAM_GAP = 0.5  # seconds between calls; 4 parallel calls drew rejections at once
SHAZAM_BACKOFF = (5.0, 15.0)  # seconds to wait before the 2nd and 3rd try of a failed call
SHAZAM_TIMEOUT = 20.0  # seconds per call; aiohttp's own default is 300


def available() -> bool:
    """The timeline needs these packages and an NVIDIA GPU (Demucs and the tagger run on CUDA)."""
    mods = ("torch", "transformers", "demucs", "beat_this", "shazamio", "librosa", "soundfile")
    if not all(importlib.util.find_spec(m) for m in mods):
        return False
    import torch
    return torch.cuda.is_available()


def load_audio(path: str, sr: int, channels: int = 1) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", str(channels), "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).T.copy()


def separate(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(voice, everything else) via Demucs htdemucs at 44.1 kHz stereo."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs").to("cuda").eval()
    wav = torch.from_numpy(stereo).to("cuda")
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        stems = apply_model(model, ((wav - mean) / std)[None], shifts=0, device="cuda", progress=False)[0] * std + mean
    voice = model.sources.index("vocals")
    rest = [i for i in range(len(model.sources)) if i != voice]
    return stems[voice].cpu().numpy(), stems[rest].sum(0).cpu().numpy()


def to_tag_rate(stereo: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.resample(stereo.mean(0), orig_sr=HQ_SR, target_sr=TAG_SR)


@functools.cache
def tagger():
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    return (ASTFeatureExtractor.from_pretrained(TAGGER),
            ASTForAudioClassification.from_pretrained(TAGGER).to("cuda").half().eval())


def score(mono: np.ndarray, starts: list[int]) -> np.ndarray:
    """Sigmoid scores [len(starts), 527] for WIN-long windows starting at these samples."""
    import torch

    fe, model = tagger()
    win, scores = int(WIN * TAG_SR), []
    for i in range(0, len(starts), 64):
        batch = [mono[s:s + win] for s in starts[i:i + 64]]
        feats = fe(batch, sampling_rate=TAG_SR, return_tensors="pt").input_values.to("cuda").half()
        with torch.no_grad():
            scores.append(torch.sigmoid(model(feats).logits.float()).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0, 527))


def tag(mono: np.ndarray, hop: float) -> tuple[np.ndarray, np.ndarray]:
    """Window centre times and sigmoid scores [windows, 527]."""
    win, step = int(WIN * TAG_SR), int(hop * TAG_SR)
    starts = list(range(0, max(1, len(mono) - win + 1), step))
    return np.array(starts) / TAG_SR + WIN / 2, score(mono, starts)


def video_cuts(media: str) -> np.ndarray:
    """Scene-cut times from ffmpeg's scene score on a 192px copy (fast, ~1s per minute)."""
    err = subprocess.run(["ffmpeg", "-v", "info", "-i", media, "-an", "-vf",
                          f"scale=192:-2,select='gt(scene,{CUT_SCENE})',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True, errors="replace", check=True).stderr
    return np.array([float(t) for t in re.findall(r"pts_time:([0-9.]+)", err)])


def frame_db(x: np.ndarray, ref: float) -> np.ndarray:
    n = int(HIT_FRAME * HQ_SR)
    m = len(x) // n
    return 20 * np.log10(np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1)) + 1e-9) - ref


def hits(rest: np.ndarray, ref: float, stretches: list[list[float]], cuts: np.ndarray) -> list[dict]:
    """Short bursts in the voice-removed audio that the 1s tagger misses: dings, pops, swishes,
    glitch ticks sitting 10-30 dB under the voice. Tuned on Nick Saraev Shorts (2026-10-20):
    - a burst jumps HIT_RISE_DB above the 2s around it and is no quieter than HIT_MIN_DB;
    - it must jump above 4 kHz too, because mouth pops and plosives that leak past Demucs are low-only;
    - under music, hi-hats and drums jump the same way, so there it must also land on a video cut
      and jump HIT_MUSIC_RISE_DB above 4 kHz;
    - off a cut it must reach HIT_OFF_CUT_MIN_DB: a talking head's desk knocks and key taps sit
      at -20 to -23 dB, while the dings measured -9 to -11."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfilt

    mono = rest.mean(0)
    full = frame_db(mono, ref)
    high = frame_db(sosfilt(butter(4, 4000, "hp", fs=HQ_SR, output="sos"), mono), ref)
    win = int(2 / HIT_FRAME) | 1
    rise_full, rise_high = full - median_filter(full, win), high - median_filter(high, win)
    found = np.where((full >= HIT_MIN_DB) & (np.maximum(rise_full, rise_high) >= HIT_RISE_DB)
                     & (rise_high >= HIT_HIGH_RISE_DB))[0]
    out = []
    for run in np.split(found, np.where(np.diff(found) > 2)[0] + 1) if len(found) else []:
        i = int(run[np.argmax(rise_high[run])])
        t = (i + 0.5) * HIT_FRAME
        on_cut = bool(len(cuts)) and float(np.abs(cuts - t).min()) <= CUT_NEAR
        if any(a <= t <= b for a, b in stretches) and not (on_cut and rise_high[i] >= HIT_MUSIC_RISE_DB):
            continue
        if not on_cut and full[i] < HIT_OFF_CUT_MIN_DB:
            continue
        out.append({"time": round(t, 2), "db": round(float(full[i]), 1), "on_cut": on_cut})
    return out


def spans(times: np.ndarray, on: np.ndarray, hop: float, join: float, min_len: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t, flag in zip(times, on):
        if not flag:
            continue
        lo, hi = t - hop / 2, t + hop / 2
        if out and lo - out[-1][1] <= join:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return [[round(max(0.0, float(a)), 1), round(float(b), 1)] for a, b in out if b - a >= min_len]


def levels(mono: np.ndarray, times: np.ndarray, ref: float) -> np.ndarray:
    """dB of each WIN window around `times`, relative to `ref` (the whole mix)."""
    half = int(WIN * TAG_SR / 2)
    return np.array([db(mono[max(0, int(t * TAG_SR) - half):int(t * TAG_SR) + half]) - ref for t in times])


def events(times, voice_scores, rest_scores, labels, hop, voice_db, rest_db) -> list[dict]:
    """Laughter and vocal sounds from the voice part, effects from the rest; one per run of a label.
    A hit also needs its part to be loud in that window (within EVENT_DB of the whole mix): the
    tagger hears "Zipper" or "Chicken" in near-silent leftovers of a talking head, 24-42 dB down,
    while real edit effects sit near the mix level."""
    out: list[dict] = []
    voice = np.zeros_like(voice_scores)
    voice[:, VOCAL_EVENT] = voice_scores[:, VOCAL_EVENT]
    fx = rest_scores.copy()
    fx[:, sorted(NOT_EFFECT)] = 0
    for t, v, f, vdb, rdb in zip(times, voice, fx, voice_db, rest_db):
        hits = []
        laugh = float(v[list(LAUGHTER)].max())
        if laugh >= LAUGH_ON and vdb >= -LAUGH_DB:
            hits.append(("Laughter", laugh))
        elif v.max() >= EVENT_ON and vdb >= -EVENT_DB:
            hits.append((labels[int(v.argmax())], float(v.max())))
        if f.max() >= EVENT_ON and rdb >= -EVENT_DB:
            hits.append((labels[int(f.argmax())], float(f.max())))
        for label, score in hits:
            last = next((e for e in reversed(out) if e["label"] == label), None)
            if last and t - last["end"] <= hop + 0.01:
                last["end"], last["score"] = round(float(t), 1), max(last["score"], round(score, 2))
            else:
                out.append({"start": round(float(t), 1), "end": round(float(t), 1), "label": label,
                            "score": round(score, 2)})
    return sorted(out, key=lambda e: e["start"])


def db(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)


def section(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return x[:, int(a * HQ_SR):int(b * HQ_SR)]


@functools.cache
def beat_tracker():
    from beat_this.inference import Audio2Beats

    return Audio2Beats(device="cuda", dbn=False)  # dbn needs madmom, not installed here


def bpm(stereo: np.ndarray) -> int | None:
    """Tempo from the median beat gap (beat_this)."""
    mono = stereo.mean(0)[: HQ_SR * 60]
    if len(mono) < HQ_SR * 4:
        return None
    beats, _ = beat_tracker()(mono, HQ_SR)
    return round(60 / float(np.median(np.diff(beats)))) if len(beats) >= 4 else None


def shazam_all(clips: list[np.ndarray], work: Path, failures: list[str]) -> list[dict | None]:
    """Recognize clips one by one; None where Shazam has no match. A failed call (rate limit,
    network) is appended to failures and counts as no match, so one bad call never hides the rest."""
    if not clips:
        return []
    import soundfile as sf
    from aiohttp_retry import ExponentialRetry
    from shazamio import HTTPClient, Shazam

    # shazamio's default retries a 429 twenty times with backoff up to 60s: minutes of silent hang.
    # Retries are ours instead (SHAZAM_BACKOFF), so they also cover a rate-limit body that is not JSON.
    client = Shazam(segment_duration_seconds=int(SHAZAM_CLIP), http_client=HTTPClient(
        retry_options=ExponentialRetry(attempts=1, statuses=set())))

    async def one(i, audio):
        if audio.shape[1] < HQ_SR * 3:
            return None
        path = work / f"shazam-{len(failures)}-{i}-{audio.shape[1]}.wav"
        sf.write(str(path), audio.T, HQ_SR)
        for wait in (*SHAZAM_BACKOFF, None):
            try:
                out = await asyncio.wait_for(client.recognize(str(path)), SHAZAM_TIMEOUT)
                break
            # A rate limit arrives as a non-JSON body (FailedDecodeJson), not only as a 429.
            except Exception as exc:  # noqa: BLE001 - shazamio raises bare aiohttp/decode errors
                if wait is None:
                    failures.append(f"{type(exc).__name__}: {exc}"[:200])
                    return None
                await asyncio.sleep(wait)
        await asyncio.sleep(SHAZAM_GAP)
        track, matches = out.get("track"), out.get("matches") or []
        if not track or not matches:
            return None
        return {"id": str(track.get("key")), "title": track.get("title"), "artist": track.get("subtitle"),
                "url": track.get("url"), "offset": float(matches[0].get("offset") or 0)}

    async def run():  # one call at a time: parallel calls get rejected, each takes ~0.3s anyway
        return [await one(i, c) for i, c in enumerate(clips)]

    return asyncio.run(run())


def identify(probes: list[tuple[float, float, bool]], mix, rest, work: Path, failures: list[str]) -> list[dict | None]:
    """Song per probe (start, end, voiced). The plain mix is asked first. Under a voice the
    voice-removed clip is asked too, and counts only when a clip CONFIRM_SHIFT later lands on
    the same song at the matching offset: Demucs leftovers fool Shazam into random matches."""
    hits = shazam_all([section(mix, a, b) for a, b, _ in probes], work, failures)
    retry = [i for i, (h, (_, _, voiced)) in enumerate(zip(hits, probes)) if not h and voiced][:SHAZAM_RETRIES]
    pairs = shazam_all([section(rest, probes[i][0] + s, probes[i][1] + s)
                        for i in retry for s in (0.0, CONFIRM_SHIFT)], work, failures)
    out = [{**h, "via": "mix"} if h else None for h in hits]
    for n, i in enumerate(retry):
        first, again = pairs[2 * n], pairs[2 * n + 1]
        if first and again and first["id"] == again["id"] and abs(again["offset"] - first["offset"] - CONFIRM_SHIFT) < 2:
            out[i] = {**first, "via": "voice removed"}
    return out


def analyze(media: str, work: Path, has_video: bool, shazam: bool = True) -> dict:
    """Full-media audio timeline. Times are absolute seconds in the media.
    shazam=False names no songs and makes zero Shazam calls (reel QA knows its own bed)."""
    print("[watch] separating voice from music on GPU (Demucs)…", file=sys.stderr)
    mix = load_audio(media, HQ_SR, channels=2)
    duration = mix.shape[1] / HQ_SR
    voice, rest = separate(mix)
    print("[watch] tagging music, laughter and sound effects (AudioSet)…", file=sys.stderr)
    hop = 0.5 if duration <= 300 else 1.0
    voice16, rest16 = to_tag_rate(voice), to_tag_rate(rest)
    times, voice_scores = tag(voice16, hop)
    _, rest_scores = tag(rest16, hop)
    ref = db(voice16 + rest16)
    labels = {int(k): v for k, v in tagger()[1].config.id2label.items()}
    music_on = np.convolve(rest_scores[:, MUSIC], np.ones(MUSIC_SMOOTH) / MUSIC_SMOOTH, mode="same") >= MUSIC_ON
    stretches = spans(times, music_on, hop, MUSIC_JOIN, MUSIC_MIN)

    music, probes = [], []
    for a, b in stretches:
        m = (times >= a) & (times <= b)
        detail = rest_scores[m][:, MUSIC_DETAIL].mean(0)
        speech = float((voice_scores[m][:, list(SPEECH)].max(1) >= 0.5).mean())
        info = {"start": a, "end": b, "speech": round(speech, 2), "bpm": bpm(section(rest, a, b)),
                "labels": [labels[MUSIC_DETAIL[i]] for i in np.argsort(detail)[::-1][:4] if detail[i] >= 0.05],
                "songs": []}
        if speech >= 0.2:
            info["under_voice_db"] = round(db(section(rest, a, b)) - db(section(voice, a, b)), 1)
        music.append(info)
        # Shorts swap songs every few seconds; long videos spread the probe budget.
        for p in np.arange(a, max(a + 0.1, b - 4), max(SHAZAM_EVERY, duration / SHAZAM_PROBES)):
            probes.append((len(music) - 1, float(p), min(b, float(p) + SHAZAM_CLIP), speech >= 0.2))
    if not shazam:
        probes = []
    if len(probes) > SHAZAM_PROBES:  # many short stretches: spread the budget across them
        step = len(probes) / SHAZAM_PROBES
        probes = [probes[int(i * step)] for i in range(SHAZAM_PROBES)]

    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] naming the music at {len(probes)} point(s) with Shazam…", file=sys.stderr)
    failures: list[str] = []
    for (k, p, _, _), hit in zip(probes, identify([q[1:] for q in probes], mix, rest, work, failures)):
        if not hit:
            continue
        songs = music[k]["songs"]
        known = next((s for s in songs if s["id"] == hit["id"]), None)
        if known:
            known["heard_at"].append(round(p, 1))
        else:
            songs.append({key: hit[key] for key in ("id", "title", "artist", "url", "via")} | {"heard_at": [round(p, 1)]})
    found = events(times, voice_scores, rest_scores, labels, hop, levels(voice16, times, ref), levels(rest16, times, ref))
    print("[watch] looking for quiet hits at video cuts…", file=sys.stderr)
    quiet = hits(rest, db(mix.mean(0)), stretches, video_cuts(media) if has_video else np.array([]))
    quiet = [h for h in quiet if not any(e["start"] - 0.5 <= h["time"] <= e["end"] + 0.5 for e in found)]
    # Name each hit from a 1s window centred on it; effect classes only (its own speech leaks in).
    named = score(rest16, [max(0, int((h["time"] - WIN / 2) * TAG_SR)) for h in quiet])
    named[:, sorted(NOT_EFFECT)] = 0
    for h, row in zip(quiet, named):
        k = int(row.argmax())
        found.append({"start": h["time"], "end": h["time"], "label": labels[k] if row[k] >= HIT_NAME_ON else "unnamed",
                      "score": round(float(row[k]), 2), "hit": {"db": h["db"], "on_cut": h["on_cut"]}})
    return {"duration": round(duration, 1), "music": music, "shazam_points": len(probes),
            "shazam_failures": failures, "events": sorted(found, key=lambda e: e["start"])}


def cached(media: str, saved: Path | None, work: Path, has_video: bool) -> dict:
    """analyze(), kept beside the cached media; recomputed when VERSION changes. A result with
    failed Shazam lookups is never kept, so the next run asks again."""
    if saved and saved.is_file():
        result = json.loads(saved.read_text(encoding="utf-8"))
        if result.get("version") == VERSION and not result.get("shazam_failures"):
            return result
    result = {"version": VERSION, **analyze(media, work, has_video)}
    if saved and not result["shazam_failures"]:
        # Atomic: an interrupted or concurrent run never leaves a half-written cache behind.
        tmp = saved.with_name(f"{saved.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        os.replace(tmp, saved)
    return result


def clock(t: float) -> str:
    t = round(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def summary(result: dict, lo: float, hi: float) -> str:
    music = [m for m in result["music"] if m["end"] >= lo and m["start"] <= hi]
    events = [e for e in result["events"] if e["end"] >= lo and e["start"] <= hi]
    covered = sum(min(m["end"], hi) - max(m["start"], lo) for m in music)
    songs = {s["id"] for m in music for s in m["songs"]}
    laughs = sum(e["label"] == "Laughter" for e in events)
    return (f"music {covered / max(hi - lo, 1e-9):.0%} of the range in {len(music)} stretch(es), "
            f"{len(songs)} song(s) named, {laughs} laugh(s), {len(events) - laughs} other sound(s)")


def report(result: dict, lo: float, hi: float) -> list[str]:
    """Markdown lines for the `## Audio` section, limited to [lo, hi] seconds."""
    music = [m for m in result["music"] if m["end"] >= lo and m["start"] <= hi]
    events = [e for e in result["events"] if e["end"] >= lo and e["start"] <= hi]
    out = ["_Demucs splits the voice from the rest; an AudioSet tagger labels 1s windows; Shazam names songs. "
           "Tagger labels are guesses: check them against the frames. A song is named only on a Shazam match._", "",
           "**Music**", ""]
    for m in music:
        bits = [f"- {clock(m['start'])} → {clock(m['end'])} ({m['end'] - m['start']:.0f}s)"]
        if m["labels"]:
            bits.append(", ".join(m["labels"]))
        if m["bpm"]:
            bits.append(f"~{m['bpm']} BPM")
        if "under_voice_db" in m:
            d = m["under_voice_db"]
            bits.append(f"under a voice, {abs(d):.0f} dB {'below' if d < 0 else 'above'} it")
        line = " · ".join(bits)
        if m["songs"]:
            line += " · " + "; ".join(
                f"**\"{s['title']}\" by {s['artist']}** (heard at {', '.join(clock(t) for t in s['heard_at'])}"
                f"{', voice removed' if s['via'] != 'mix' else ''}) {s['url']}" for s in m["songs"])
        else:
            line += " · no Shazam match (stock, library or custom music, or too quiet under the voice)"
        out.append(line)
    if not music:
        out.append("- No music found.")
    if result.get("shazam_failures"):
        f = result["shazam_failures"]
        out += ["", f"- **Shazam failed on {len(f)} lookup(s)** (first: {f[0]}). Songs there may be missing; "
                    "re-run in a few minutes (this result was not cached, so the next run asks again)."]
    laughs = [e for e in events if e["label"] == "Laughter"]
    other = [e for e in events if e["label"] != "Laughter"]
    def span(e):
        return clock(e["start"]) + (f"-{clock(e['end'])}" if e["end"] > e["start"] else "")

    out += ["", f"**Laughter:** {', '.join(span(e) for e in laughs) if laughs else 'none found'}", "",
            "**Sound effects and other sounds**", ""]
    def line(e):
        text = f"- {span(e)} {e['label']} ({e['score']:.2f})"
        if "hit" in e:  # tenths of a second: a hit is lined up against a cut or a caption word
            text = f"- {clock(int(e['start']))}.{int(e['start'] * 10) % 10} {e['label']} ({e['score']:.2f})"
            text += (f" · quiet hit, {abs(e['hit']['db']):.0f} dB {'under' if e['hit']['db'] < 0 else 'over'} the mix average"
                     + (", on a video cut" if e["hit"]["on_cut"] else ""))
        return text

    out += [line(e) for e in other[:EVENT_LINES]] or ["- None found."]
    if len(other) > EVENT_LINES:
        out.append(f"- … {len(other) - EVENT_LINES} more; re-run with --start/--end to list a section.")
    return out


if __name__ == "__main__":
    import tempfile

    if len(sys.argv) != 2:
        raise SystemExit("usage: audio.py <media>")
    print(json.dumps(analyze(sys.argv[1], Path(tempfile.mkdtemp(prefix="watch-audio-")), has_video=True), indent=2))
