"""Generate keyframes for selected shots with face consistency using Stable Diffusion Img2Img.

This script re‑generates only the shots you specify (e.g. s05) while keeping the three main
characters' faces identical across the whole story. It uses the reference sheet produced
by `pipeline/keyframes.py` (saved as `outputs/keyframes/ref_sheet.png`) as the conditioning
image.

Requirements (install once):
    pip install diffusers[torch] transformers accelerate scipy

Usage example:
    .venv/bin/python3 pipeline/consistent_keyframes.py \
        --shots s05 s09 \
        --reference outputs/keyframes/ref_sheet.png \
        --output_dir outputs/keyframes

The script writes PNG files `outputs/keyframes/s05.png`, `.../s09.png` etc.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_storyboard(path: Path = Path("storyboard.json")) -> dict:
    return json.loads(path.read_text())

def build_prompt(shot: dict, sb: dict) -> str:
    """Return a concise prompt for Img2Img.
    We only keep the shot's own image_prompt to avoid token overflow.
    """
    return shot["image_prompt"].strip()

# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_img2img(
    prompt: str,
    init_image_path: Path,
    seed: int,
    steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.4,
    output_path: Path = None,
) -> None:
    """Run Stable Diffusion Img2Img and write the result to `output_path`.

    Args:
        prompt: textual prompt.
        init_image_path: path to the reference image (the face sheet).
        seed: deterministic seed for reproducibility.
        steps: diffusion steps.
        guidance_scale: classifier‑free guidance weight.
        strength: how much to transform the init image (0 = keep init, 1 = ignore).
        output_path: where to save the PNG.
    """
    torch.manual_seed(seed)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # pipe.enable_xformers_memory_efficient_attention()  # Disabled: xformers not installed


    init_image = Image.open(init_image_path).convert("RGB")
    init_image = init_image.resize((1024, 576))  # match keyframe size
    # NOTE: Keep init_image as a PIL.Image for diffusers; no extra conversion needed.
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )
    image = result.images[0]
    if output_path:
        image.save(output_path)
        print(f"✅ {output_path.name} saved")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate selected keyframes with face consistency")
    parser.add_argument(
        "--shots",
        nargs="+",
        required=True,
        help="Shot IDs to regenerate (e.g. s05 s09) or 'all' for every shot. Use 'not' followed by IDs to exclude them, e.g. '--shots all not s05 s09'.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference sheet PNG (outputs/keyframes/ref_sheet.png)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/keyframes"),
        help="Directory where regenerated PNGs will be written",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Determine which shots to process
    # ---------------------------------------------------------------------------
    sb = load_storyboard()
    all_shot_ids = {shot["id"] for shot in sb["shots"]}
    # args.shots is a list of tokens, e.g. ['all', 'not', 's05', 's09']
    if "all" in args.shots:
        # start with everything
        selected = set(all_shot_ids)
        if "not" in args.shots:
            not_index = args.shots.index("not")
            exclude = set(args.shots[not_index + 1 :])
            selected -= exclude
    else:
        # explicit list
        selected = set(args.shots)

    # ---------------------------------------------------------------------------
    # Regenerate each selected shot using Img2Img with the reference sheet
    # ---------------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for shot in sb["shots"]:
        sid = shot["id"]
        if sid not in selected:
            continue
        prompt = build_prompt(shot, sb)
        seed = shot.get("seed", 42)
        out_path = args.output_dir / f"{sid}.png"
        generate_img2img(
            prompt=prompt,
            init_image_path=args.reference,
            seed=seed,
            steps=30,
            guidance_scale=7.5,
            strength=0.4,
            output_path=out_path,
        )

if __name__ == "__main__":
    main()

