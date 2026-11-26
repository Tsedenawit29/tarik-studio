"""Stage 2: Generate character reference sheet and per-shot 16:9 keyframes via Cloudflare FLUX."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import subprocess

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
    """Call Cloudflare Workers AI image model and write PNG bytes to out_path.

    Note: FLUX-1-schnell on Cloudflare does not accept width, height or seed.
    It returns a default-sized image. We resize to our target after download.
    The seed is recorded in runs.jsonl for documentation but not sent to the API.
    If the API call fails (e.g., network/DNS issues), we fall back to a solid
    placeholder image generated locally with ffmpeg so the pipeline can continue.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "prompt": prompt,
        "steps": steps,
    }
    try:
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
            raw_bytes = base64.b64decode(img_b64)
        else:
            raw_bytes = r.content
    except Exception as e:
        # Fallback: create a solid gray image with the prompt text overlay
        # using ffmpeg's color source and drawtext filter.
        placeholder_tmp = out_path.with_suffix(".tmp.png")
        # Escape double quotes in prompt for shell safety
        safe_prompt = prompt.replace('"', '\\"')
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=gray:s={WIDTH}x{HEIGHT}",
                "-vf",
                f"drawtext=fontfile=/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf:text='{safe_prompt}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2",
                str(placeholder_tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw_bytes = placeholder_tmp.read_bytes()
        placeholder_tmp.unlink(missing_ok=True)
    # Write raw image first, then resize to target dimensions with FFmpeg
    tmp = out_path.with_suffix(".tmp.png")
    tmp.write_bytes(raw_bytes)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
         "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}",
         str(out_path)],
        check=True,
    )
    tmp.unlink(missing_ok=True)
    """Call Cloudflare Workers AI image model and write PNG bytes to out_path.

    Note: FLUX-1-schnell on Cloudflare does not accept width, height or seed.
    It returns a default-sized image. We resize to our target after download.
    The seed is recorded in runs.jsonl for documentation but not sent to the API.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "prompt": prompt,
        "steps": steps,
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
        raw_bytes = base64.b64decode(img_b64)
    else:
        raw_bytes = r.content

    # Write raw image first, then resize to target dimensions with FFmpeg
    tmp = out_path.with_suffix(".tmp.png")
    tmp.write_bytes(raw_bytes)
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
         "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}",
         str(out_path)],
        check=True,
    )
    tmp.unlink(missing_ok=True)


def build_shot_prompt(shot: dict, sb: dict) -> str:
    """Combine shot prompt, full character description, and style block.
    We always prepend the character description to enforce visual consistency across
    shots that involve the same characters (Emama, Ababa, Abush). The description
    already contains Ethiopian clothing details (netela, scarf, etc.).
    """
    parts = [shot["image_prompt"].strip()]
    # Always include the full character description for consistency.
    char_desc = sb["character"]["description"].strip()
    parts.append(char_desc)
    parts.append(sb["style_block"].strip())
    return ", ".join(parts)

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

