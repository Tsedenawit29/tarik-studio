"""Stage 1: Generate per-shot narration WAVs via edge-tts + ffmpeg (with dual Amharic voices & optional laugh SFX)."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import edge_tts

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

NARRATION_DIR = OUTPUTS / "narration"
SFX_DIR = Path("assets/sfx")

# Applied only to TTS input text; storyboard.json remains unchanged.
PRONUNCIATION = {
    "ብራና": "በራና",
    "ዓባይ": "አብባይ",
}

# Shot-level voice overrides: Ababa (s03-s04) uses male voice; remaining shots use female voice
SHOT_VOICES: dict[str, str] = {
    "s03": "am-ET-AmehaNeural",
    "s04": "am-ET-AmehaNeural",
}

# Optional local laugh SFX clips mixed gently under the voice on comic moments
SHOT_LAUGH_SFX: dict[str, Path] = {
    "s07": SFX_DIR / "laugh_child.wav",
    "s09": SFX_DIR / "laugh_family.wav",
}


def spoken(text: str) -> str:
    for written, said in PRONUNCIATION.items():
        text = text.replace(written, said)
    return text


def _prosody_for_shot(sid: str, style: str) -> tuple[str, str]:
    """Tune speech rate and pitch slightly per character/moment so delivery feels less flat."""
    if sid in ("s03", "s04"):  # Ababa: slightly slower, grounded
        return "-6%", "-2Hz"
    if sid == "s07":  # Abush's playful "እነ ነብር ነሽ!" line: slightly brighter/faster
        return "+3%", "+2Hz"
    if sid == "s10":  # Closing title punchline: warm, deliberate
        return "-8%", "-1Hz"
    s = (style or "").lower()
    if "lighthearted" in s or "playful" in s:
        return "+0%", "+1Hz"
    return "-4%", "0Hz"


async def _synth_mp3(text: str, voice: str, rate: str, pitch: str, mp3_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(mp3_path))


def _mix_laugh_if_present(wav_path: Path, sfx_path: Path | None) -> None:
    if not sfx_path or not sfx_path.exists() or not shutil.which("ffmpeg"):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        mixed_path = Path(tmpdir) / "mixed.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-i",
            str(sfx_path),
            "-filter_complex",
            "[1:a]adelay=1800|1800,volume=0.45[laugh];[0:a][laugh]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map",
            "[out]",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(mixed_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(str(mixed_path), wav_path)


def synthesize_shot(
    sid: str,
    text: str,
    style: str,
    voice: str,
    out_path: Path,
    sfx_path: Path | None = None,
) -> None:
    """Synthesize speech with edge-tts, convert to 24kHz mono WAV via ffmpeg, and optionally mix laugh SFX."""
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    rate, pitch = _prosody_for_shot(sid, style)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mp3 = Path(tmpdir) / "shot.mp3"
        tmp_wav = Path(tmpdir) / "shot.wav"

        asyncio.run(_synth_mp3(text=text, voice=voice, rate=rate, pitch=pitch, mp3_path=tmp_mp3))

        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_mp3), "-ar", "24000", "-ac", "1", str(tmp_wav)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            shutil.move(str(tmp_wav), out_path)
        else:
            fallback_mp3 = out_path.with_suffix(".mp3")
            shutil.move(str(tmp_mp3), fallback_mp3)
            out_path = fallback_mp3

    _mix_laugh_if_present(out_path, sfx_path)

    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"Generated audio missing or too small: {out_path}")


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    default_voice = os.environ.get("EDGE_TTS_VOICE", "am-ET-AmehaNeural")
    style = sb.get("narration_style", "warm, lighthearted, affectionate, storytelling")
    language = sb["language"]

    requested_shots = {arg for arg in sys.argv[1:] if not arg.startswith("--")}
    dry_run = "--dry-run" in sys.argv[1:]

    for shot in sb["shots"]:
        sid = shot["id"]
        if requested_shots and str(sid) not in requested_shots:
            continue

        voice = SHOT_VOICES.get(sid, default_voice)
        sfx_path = SHOT_LAUGH_SFX.get(sid)
        sfx_mtime = str(sfx_path.stat().st_mtime_ns) if sfx_path and sfx_path.exists() else "none"
        model = f"edge-tts:{voice}"

        raw_text = shot["narration"]["text"]
        text = spoken(raw_text)
        key = sha("narration", sid, text, style, voice, language, model, sfx_mtime)
        out = NARRATION_DIR / f"{sid}.wav"

        if dry_run:
            changed = text != raw_text
            print(f"[{sid}] voice={voice} sfx={sfx_path if sfx_mtime != 'none' else None} changed={changed} | spoken={text}")
            continue

        def produce(path: Path, _sid=sid, _text=text, _voice=voice, _sfx=sfx_path) -> None:
            logged_call(
                stage="narration",
                shot=_sid,
                model=model,
                params={
                    "voice": _voice,
                    "style": style,
                    "language": language,
                    "chars": len(_text),
                },
                fn=lambda: synthesize_shot(_sid, _text, style, _voice, path, sfx_path=_sfx),
            )

        if cached("narration", sid, key, out, produce):
            time.sleep(0.3)


if __name__ == "__main__":
    main()