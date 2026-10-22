# /watch

Let your AI agent watch a video.
Give it a YouTube, TikTok or Instagram link, or a file on your disk.
It gets timestamped frames, a transcript, and (optionally) the music, sound effects and on-screen text.
Then it answers your question about the video.

Works as a skill in Claude Code and in Codex or any other agent that supports [Agent Skills](https://agentskills.io).

## Install (30 seconds)

You need `git`, Python 3.10+, `ffmpeg` and `yt-dlp` ([install them](#requirements)).

**Claude Code** (Mac, Linux, Windows):

```bash
git clone https://github.com/RyanCollinsAI/watch ~/.claude/skills/watch
```

**Codex and other agents:** clone into the agent's skills folder, for example:

```bash
git clone https://github.com/RyanCollinsAI/watch ~/.agents/skills/watch
```

Start a new session.
Then type:

```text
/watch https://www.youtube.com/watch?v=dQw4w9WgXcQ what is this video about?
```

In an agent without slash commands, say "use the watch skill on this video: <link>".

To update, run `git pull` inside the skill folder.

## What you get

- **Frames:** scene changes, duplicates dropped, plus a frame wherever the speaker says "look at this" or "as you can see", plus gap-fill frames so no stretch goes unseen.
- **Contact sheets:** 9 frames per image with a timestamp on each, so the agent reads a whole video in a few images.
- **Transcript:** YouTube captions first, then local speech-to-text (faster-whisper on CPU or GPU, or WhisperX), or Groq/OpenAI if you give a key.
- **Report:** title, chapters, pacing (cuts per minute), and a transcript with timestamps.
- **Cache:** the download, transcript and audio analysis are kept for 14 days, so follow-up questions are instant.
- **Optional audio timeline** (NVIDIA GPU): songs by name (Shazam), BPM, how loud the music sits under the voice, sound effects, laughter.
- **Optional on-screen text:** titles, burned-in captions and slide labels read with OCR.

## Requirements

Required:

| Tool | Mac | Linux (Debian/Ubuntu) | Windows |
|---|---|---|---|
| Python 3.10+ | `brew install python` | `sudo apt install python3` | [python.org](https://www.python.org/downloads/windows/) |
| ffmpeg | `brew install ffmpeg` | `sudo apt install ffmpeg` | `winget install Gyan.FFmpeg` |
| yt-dlp (latest) | `brew install yt-dlp` | `pipx install yt-dlp` | `winget install yt-dlp.yt-dlp` |

Keep yt-dlp up to date: sites break old versions.
YouTube also wants [Deno](https://docs.deno.com/runtime/getting_started/installation/) for yt-dlp.

Check your setup any time:

```bash
python3 ~/.claude/skills/watch/scripts/setup.py --json
```

On Windows use `py` instead of `python3`.

## Speech-to-text options

`/watch` first uses the video's own captions.
If there are none, it uses the backend you pick in `~/.config/watch/.env`:

| `WATCH_WHISPER_BACKEND` | What it is |
|---|---|
| `fasterwhisper` | Local. `pip install faster-whisper`. Uses an NVIDIA GPU if there is one (model `large-v3-turbo`), else the CPU (model `small`, slower). Set `WATCH_FW_MODEL` to change the model. With a GPU, captions are skipped because local Whisper is better than auto captions. |
| `whisperx` | Local, CPU. Run `setup.py --install-whisperx` once. |
| `groq` / `openai` | Cloud. Needs `GROQ_API_KEY` or `OPENAI_API_KEY` in the same file. |
| `none` | Captions only. |

Or set `GEMINI_API_KEY` and Google's video model watches the video for you (`--engine gemini`).

Settings and keys live in `~/.config/watch/.env`.
Never commit that file.

## Optional extras

```bash
pip install -r requirements-extras.txt
```

Install the right PyTorch build for your machine first ([pytorch.org](https://pytorch.org/get-started/locally/)).
The audio timeline needs an NVIDIA GPU.
On-screen text works on CPU and GPU.
If the extras are missing, `/watch` skips those parts and says so on stderr.
Use `--no-audio` and `--no-ocr` to skip them on purpose.
`--no-audio` also keeps everything off Shazam; only short 12-second clips ever leave your machine for song lookup.

## Useful flags

| Flag | What it does |
|---|---|
| `--question TEXT` | The question you want answered |
| `--start 1:00 --end 2:30` | Look closer at one section |
| `--timestamps 4:32,7:10` | Pin frames at exact moments |
| `--detail transcript\|efficient\|balanced\|token-burner` | Fewer or more frames |
| `--sub-lang en` | Use captions even when a local Whisper is set up |
| `--cookies-from-browser chrome` | For videos behind a login |

## Troubleshooting

| Problem | Fix |
|---|---|
| Download fails with 403 | `pip install -U yt-dlp`, then retry |
| 429 from YouTube | Wait, or use a local Whisper backend (it skips captions) |
| Login wall or Instagram block | Add `--cookies-from-browser chrome` |
| "audio timeline skipped" | Expected without an NVIDIA GPU and the extras |
| Python opens the Microsoft Store (Windows) | Use `py` instead of `python3` |

## Development

```bash
pip install pytest
pytest -q
```

Tests use generated video clips and make no network calls.

## Credit

This is a fork of [bradautomates/claude-video](https://github.com/bradautomates/claude-video) (MIT) by Bradley Bonanno, synced to upstream 0.3.2.
Thank you to Bradley for the original, and to the fork authors whose ideas were ported: [abe238/claude-video-plus](https://github.com/abe238/claude-video-plus) (process timeouts, H.264-first downloads, speaker labels in captions).

Added in this fork: contact sheets, spoken-cue and gap-fill frames, near-black frame drop, chapters and pacing, media and transcript cache, faster-whisper (GPU or CPU), the audio timeline, on-screen text, and a flat install-by-clone layout.

MIT licensed, see [LICENSE](LICENSE).
Made by [Ryan Collins](https://github.com/RyanCollinsAI).
