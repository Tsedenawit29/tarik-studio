"""Dry-run planner: show which outputs are cached and which would regenerate."""
from __future__ import annotations

import os
from pathlib import Path

from common import OUTPUTS, load_env, load_storyboard, sha
from keyframes import HEIGHT, WIDTH, build_shot_prompt


def _status(out: Path, key: str) -> str:
    stamp = out.with_suffix(out.suffix + ".key")
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return f"CACHED ({key})"
    if out.exists():
        return f"STALE  ({key})"
    return f"BUILD  ({key})"


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    tts_model = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
    style = sb.get("narration_style", "calm, warm, unhurried")
    voice = sb.get("narration_voice", os.environ.get("GEMINI_TTS_VOICE", "Charon"))
    lang = sb["language"]

    cf_model = os.environ.get("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")

    print("=== PLAN ===")
    for shot in sb["shots"]:
        sid = shot["id"]
        n_key = sha("narration", sid, shot["narration"]["text"], style, voice, lang, tts_model)
        n_out = OUTPUTS / "narration" / f"{sid}.wav"

        k_prompt = build_shot_prompt(shot, sb)
        k_key = sha("keyframe", sid, k_prompt, shot["seed"], cf_model, WIDTH, HEIGHT)
        k_out = OUTPUTS / "keyframes" / f"{sid}.png"

        wan_clip = OUTPUTS / "shots" / "wan" / f"{sid}.mp4"
        s_mode = "wan" if (shot.get("motion") == "wan" and wan_clip.exists()) else f"kenburns:{shot.get('kenburns', 'push_in')}"
        s_out = OUTPUTS / "shots" / f"{sid}.mp4"
        s_state = "EXISTS" if s_out.exists() else "BUILD"

        print(f"  {sid}: narration={_status(n_out, n_key)} | keyframe={_status(k_out, k_key)} | shot={s_state} ({s_mode})")


if __name__ == "__main__":
    main()

