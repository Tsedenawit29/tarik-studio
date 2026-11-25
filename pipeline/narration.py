"""Stage 1: Generate per-shot narration WAVs via Gemini TTS with per-shot voice & laugh cues."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

NARRATION_DIR = OUTPUTS / "narration"

# Applied only to TTS input text; storyboard.json remains unchanged.
PRONUNCIATION = {
    "ብራና": "በራና",
    "ዓባይ": "አብባይ",
}

# Shot-level voice overrides so Ababa (s03-s04) contrasts with Ene / narration
SHOT_VOICES: dict[str, str] = {
    "s03": "Charon",
    "s04": "Charon",
}

# Shot-level expressive cues appended to style direction without mutating storyboard.json
SHOT_LAUGH_CUES: dict[str, str] = {
    "s07": "playful laugh right after saying 'እነ ነብር ነሽ!'",
    "s09": "warm shared family laughter at the end of the line",
}


def spoken(text: str) -> str:
    for written, said in PRONUNCIATION.items():
        text = text.replace(written, said)
    return text


def _build_direction(style: str, laugh_cue: str | None) -> str:
    parts = [p.strip() for p in (style or "").split(",") if p.strip()]
    if laugh_cue:
        parts.append(laugh_cue.strip())
    return ", ".join(parts)


def _pcm_to_wav(pcm_bytes: bytes, out_path: Path, sample_rate: int = 24000) -> None:
    if pcm_bytes[:4] == b"RIFF" and pcm_bytes[8:12] == b"WAVE":
        out_path.write_bytes(pcm_bytes)
        return
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _normalize_wav(wav_path: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-ar", "24000", "-ac", "1", str(wav_path.with_suffix(".norm.wav"))],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        norm = wav_path.with_suffix(".norm.wav")
        if norm.exists() and norm.stat().st_size > 1000:
            norm.replace(wav_path)
    except FileNotFoundError:
        pass


def synthesize_shot(
    client: genai.Client,
    text: str,
    style: str,
    voice: str,
    model: str,
    out_path: Path,
    laugh_cue: str | None = None,
) -> None:
    """Synthesize a single shot's narration WAV via google-genai."""
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    direction = _build_direction(style, laugh_cue)
    prompt_text = f"({direction}) {text}" if direction else text

    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )

    audio_bytes = b""
    for cand in response.candidates or []:
        if not cand.content or not cand.content.parts:
            continue
        for part in cand.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                audio_bytes = inline.data
                break
        if audio_bytes:
            break

    if len(audio_bytes) < 500:
        raise RuntimeError(f"Returned audio too short ({len(audio_bytes)} bytes) for {out_path.name}")

    _pcm_to_wav(audio_bytes, out_path)
    _normalize_wav(out_path)
    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"Output WAV missing or too small: {out_path}")


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in .env")

    model = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
    default_voice = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")
    style = sb.get("narration_style", "calm, warm, unhurried")
    language = sb["language"]

    requested_shots = {arg for arg in sys.argv[1:] if not arg.startswith("--")}
    dry_run = "--dry-run" in sys.argv[1:]

    client = genai.Client(api_key=api_key)

    for shot in sb["shots"]:
        sid = shot["id"]
        if requested_shots and str(sid) not in requested_shots:
            continue

        voice = SHOT_VOICES.get(sid, default_voice)
        laugh_cue = SHOT_LAUGH_CUES.get(sid)
        raw_text = shot["narration"]["text"]
        text = spoken(raw_text)
        key = sha("narration", sid, text, style, voice, language, model, laugh_cue or "")
        out = NARRATION_DIR / f"{sid}.wav"

        if dry_run:
            changed = text != raw_text
            print(f"[{sid}] voice={voice} laugh={laugh_cue} changed={changed} | spoken={text}")
            continue

        def produce(path: Path, _sid=sid, _text=text, _voice=voice, _laugh=laugh_cue) -> None:
            logged_call(
                stage="narration",
                shot=_sid,
                model=model,
                params={
                    "voice": _voice,
                    "style": style,
                    "language": language,
                    "laugh_cue": _laugh,
                    "chars": len(_text),
                },
                fn=lambda: synthesize_shot(client, _text, style, _voice, model, path, laugh_cue=_laugh),
            )

        if cached("narration", sid, key, out, produce):
            time.sleep(1.5)


if __name__ == "__main__":
    main()