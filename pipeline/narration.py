"""Stage 1: Generate per-shot narration WAVs via Gemini TTS (Interactions API)."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

NARRATION_DIR = OUTPUTS / "narration"
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def _extract_audio_bytes(resp_json: dict) -> bytes:
    """Extract base64 WAV audio from a Gemini Interactions API response."""
    steps = resp_json.get("steps", [])
    for step in reversed(steps):
        if step.get("type") == "model_output":
            for item in reversed(step.get("content", [])):
                if item.get("type") == "audio" and item.get("data"):
                    return base64.b64decode(item["data"])
    # Fallback for top-level outputs if returned in that shape
    for item in reversed(resp_json.get("outputs", [])):
        if item.get("type") == "audio" and item.get("data"):
            return base64.b64decode(item["data"])
    raise RuntimeError(f"No audio payload found in Gemini TTS response: {list(resp_json.keys())}")


def synthesize_shot(text: str, style: str, voice: str, model: str, api_key: str, out_path: Path) -> None:
    """Call Gemini TTS Interactions API and write a WAV file to out_path."""
    payload = {
        "model": model,
        "input": [{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": style,
                }],
            }],
        }],
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": [{"voice": voice}],
        },
    }
    r = requests.post(
        INTERACTIONS_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    audio_bytes = _extract_audio_bytes(r.json())
    if len(audio_bytes) < 1000:
        raise RuntimeError(f"Returned audio too short ({len(audio_bytes)} bytes)")
    out_path.write_bytes(audio_bytes)


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in .env")

    style = sb.get("narration_style", "calm, warm, unhurried")
    voice = sb.get("narration_voice", os.environ.get("GEMINI_TTS_VOICE", "Charon"))
    language = sb["language"]

    for shot in sb["shots"]:
        sid = shot["id"]
        text = shot["narration"]["text"]
        key = sha("narration", sid, text, style, voice, language, model)
        out = NARRATION_DIR / f"{sid}.wav"

        def produce(path: Path, _sid=sid, _text=text) -> None:
            logged_call(
                stage="narration",
                shot=_sid,
                model=model,
                params={"voice": voice, "style": style, "language": language, "chars": len(_text)},
                fn=lambda: synthesize_shot(_text, style, voice, model, api_key, path),
            )

        cached("narration", sid, key, out, produce)


if __name__ == "__main__":
    main()

