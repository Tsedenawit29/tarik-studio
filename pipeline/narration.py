"""Stage 1: Generate per-shot narration WAVs via Gemini TTS + ffmpeg (with character voices & laugh SFX)."""
from __future__ import annotations

import os
import shutil
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
SFX_DIR = Path("assets/sfx")
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")

# Applied only to TTS input text; storyboard.json remains unchanged.
PRONUNCIATION = {
    "ብራና": "በራና",
    "ዓባይ": "አብባይ",
}

# Map characters to Gemini prebuilt voices
CHARACTER_VOICES: dict[str, str] = {
    "narrator": "Aoede",
    "emama_ene": "Kore",
    "ababa": "Fenrir",
    "abush": "Puck",
}

# Shot-level speaker mapping kept in code so storyboard.json stays schema-valid
SHOT_SPEAKERS: dict[str, str] = {
    "s03": "ababa",
    "s04": "emama_ene",
    "s06": "emama_ene",
    "s07": "abush",
    "s10": "abush",
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


def _style_instruction(sid: str, speaker: str, style: str) -> str:
    """Guide Gemini TTS delivery per character and dramatic moment."""
    base = f"Read this Amharic line naturally in a {style} tone."
    if speaker == "ababa" or sid in ("s03", "s04"):
        return f"{base} Speak as a 72-year-old Ethiopian grandfather: grounded, slightly tired or firm, yet caring."
    if speaker == "emama_ene" or sid in ("s06",):
        return f"{base} Speak as a 60-year-old Ethiopian grandmother: warm, quick, affectionate, and gently conspiratorial."
    if speaker == "abush" or sid in ("s07", "s10"):
        return f"{base} Speak as an energetic, joyful 8-year-old Ethiopian boy bursting with playful excitement and amusement."
    return f"{base} Speak as a warm, vivid Ethiopian storyteller."


def _pcm_to_wav(pcm_bytes: bytes, wav_path: Path, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> None:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _mix_laugh_if_present(wav_path: Path, sfx_path: Path | None, delay_ms: int = 800) -> None:
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
            f"[1:a]adelay={delay_ms}|{delay_ms},volume=0.45[laugh];[0:a][laugh]amix=inputs=2:duration=first:dropout_transition=2[out]",
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


def _synth_gemini_pcm(client: genai.Client, prompt_text: str, voice_name: str) -> bytes:
    """Generate PCM audio using Gemini TTS via the v1beta REST endpoint.
    Returns raw PCM bytes decoded from base64.
    """
    import base64, json, requests
    # Construct payload according to Gemini TTS beta spec
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            }
        }
    }
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent"
    resp = requests.post(endpoint, params={"key": api_key}, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini TTS request failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # Navigate to audio inline data
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                audio_b64 = part["inlineData"].get("data")
                if audio_b64:
                    return base64.b64decode(audio_b64)
    raise RuntimeError("Gemini TTS returned no audio data.")
    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )

    for candidate in response.candidates or []:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data

    raise RuntimeError("Gemini TTS returned no audio data.")


def synthesize_shot(
    client: genai.Client,
    sid: str,
    speaker: str,
    text: str,
    style: str,
    voice_name: str,
    out_path: Path,
    sfx_path: Path | None = None,
) -> None:
    """Synthesize speech with Gemini TTS, write 24kHz mono WAV, and optionally mix laugh SFX."""
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    instruction = _style_instruction(sid, speaker, style)
    prompt_payload = f"{instruction}\n\nText:\n{text}"

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = Path(tmpdir) / "raw.wav"
        pcm_bytes = _synth_gemini_pcm(client=client, prompt_text=prompt_payload, voice_name=voice_name)
        _pcm_to_wav(pcm_bytes=pcm_bytes, wav_path=raw_wav, sample_rate=24000, channels=1, sample_width=2)

        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_wav), "-ar", "24000", "-ac", "1", str(out_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.move(str(raw_wav), out_path)

    _mix_laugh_if_present(out_path, sfx_path)

    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"Generated audio missing or too small: {out_path}")


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")

    client = genai.Client(api_key=api_key)
    style = sb.get("narration_style", "warm, vivid, expressive, Ethiopian family storytelling")
    language = sb["language"]

    requested_shots = {arg for arg in sys.argv[1:] if not arg.startswith("--")}
    dry_run = "--dry-run" in sys.argv[1:]

    for shot in sb["shots"]:
        sid = shot["id"]
        if requested_shots and str(sid) not in requested_shots:
            continue

        speaker = SHOT_SPEAKERS.get(sid, "narrator")
        voice_name = CHARACTER_VOICES["emama_ene"]

        sfx_path = SHOT_LAUGH_SFX.get(sid)
        sfx_mtime = str(sfx_path.stat().st_mtime_ns) if sfx_path and sfx_path.exists() else "none"
        model_id = f"gemini-tts:{GEMINI_TTS_MODEL}:{voice_name}"

        raw_text = shot["narration"]["text"]
        text = spoken(raw_text)
        key = sha("narration", sid, speaker, text, style, voice_name, language, model_id, sfx_mtime)
        out = NARRATION_DIR / f"{sid}.wav"

        if dry_run:
            changed = text != raw_text
            print(
                f"[{sid}] speaker={speaker} voice={voice_name} sfx={sfx_path if sfx_mtime != 'none' else None} "
                f"changed={changed} | spoken={text}"
            )
            continue

        def produce(path: Path, _sid=sid, _spk=speaker, _text=text, _voice=voice_name, _sfx=sfx_path) -> None:
            logged_call(
                stage="narration",
                shot=_sid,
                model=model_id,
                params={
                    "speaker": _spk,
                    "voice": _voice,
                    "style": style,
                    "language": language,
                    "chars": len(_text),
                },
                fn=lambda: synthesize_shot(
                    client=client,
                    sid=_sid,
                    speaker=_spk,
                    text=_text,
                    style=style,
                    voice_name=_voice,
                    out_path=path,
                    sfx_path=_sfx,
                ),
            )

        if cached("narration", sid, key, out, produce):
            time.sleep(0.4)


if __name__ == "__main__":
    main()