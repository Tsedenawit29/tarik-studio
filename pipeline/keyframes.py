"""Stage 2: Generate character reference sheet and per-shot 16:9 keyframes via Cloudflare FLUX."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

KEYFRAMES_DIR = OUTPUTS / "keyframes"
WIDTH = 1024
HEIGHT = 576  # 16:9, multiples of 64


def generate_image(
    prompt: str,
    seed: int,
    model: str,
    account_id: str,
    api_token: str,
    out_path: Path,
    steps: int = 4,
) -> None:
    """Call Cloudflare Workers AI image model and write PNG bytes to out_path."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "width": WIDTH,
        "height": HEIGHT,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    if "application/json" in content_type:
        data = r.json()
        if not data.get("success", True):
            raise RuntimeError(f"Cloudflare API error: {data.get('errors')}")
        img_b64 = data["result"]["image"]
        out_path.write_bytes(base64.b64decode(img_b64))
    else:
        out_path.write_bytes(r.content)


def build_shot_prompt(shot: dict, sb: dict) -> str:
    """Combine shot prompt, character description (when present), and style_block."""
    parts = [shot["image_prompt"].strip()]
    char_desc = sb["character"]["description"].strip()
    lower_prompt = shot["image_prompt"].lower()
    if any(tok in lower_prompt for tok in ("scholar", "monk", "gabi", "silhouette")):
        parts.append(char_desc)
    parts.append(sb["style_block"].strip())
    return ", ".join(parts)


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    model = os.environ.get("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
    if not (account_id and api_token):
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set in .env")

    # Stage 2a: Reference sheet for character + visual style consistency
    char = sb["character"]
    ref_seed = char.get("seed", 42)
    ref_prompt = (
        f"Character and style reference sheet, three views from behind and side silhouette: "
        f"{char['description']}, {sb['style_block']}"
    )
    ref_key = sha("reference", ref_prompt, ref_seed, model, WIDTH, HEIGHT)
    ref_out = KEYFRAMES_DIR / "ref_sheet.png"

    def produce_ref(path: Path) -> None:
        logged_call(
            stage="keyframes",
            shot="ref_sheet",
            model=model,
            seed=ref_seed,
            params={"width": WIDTH, "height": HEIGHT, "kind": "reference"},
            fn=lambda: generate_image(ref_prompt, ref_seed, model, account_id, api_token, path),
        )

    cached("keyframes", "ref_sheet", ref_key, ref_out, produce_ref)

    # Stage 2b: Per-shot keyframes
    for shot in sb["shots"]:
        sid = shot["id"]
        prompt = build_shot_prompt(shot, sb)
        seed = shot["seed"]
        key = sha("keyframe", sid, prompt, seed, model, WIDTH, HEIGHT)
        out = KEYFRAMES_DIR / f"{sid}.png"

        def produce_shot(path: Path, _sid=sid, _prompt=prompt, _seed=seed) -> None:
            logged_call(
                stage="keyframes",
                shot=_sid,
                model=model,
                seed=_seed,
                params={"width": WIDTH, "height": HEIGHT, "shot_size": shot["shot_size"]},
                fn=lambda: generate_image(_prompt, _seed, model, account_id, api_token, path),
            )

        cached("keyframes", sid, key, out, produce_shot)


if __name__ == "__main__":
    main()

