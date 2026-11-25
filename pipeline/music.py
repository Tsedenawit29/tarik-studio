"""Stage 4: Pick up ACE-Step 1.5 score from Kaggle, or generate a quiet ambient placeholder."""
from __future__ import annotations

import subprocess
from pathlib import Path

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

MUSIC_DIR = OUTPUTS / "music"
CHOSEN = MUSIC_DIR / "chosen.wav"
PLACEHOLDER = MUSIC_DIR / "placeholder.wav"


def generate_placeholder(duration_s: float, out_path: Path) -> None:
    """Generate a subtle low warm drone placeholder so assembly/radio-cut works before Kaggle."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency=110:sample_rate=44100:duration={duration_s:.2f}",
        "-f", "lavfi", "-i", f"sine=frequency=164.81:sample_rate=44100:duration={duration_s:.2f}",
        "-filter_complex",
        f"[0:a][1:a]amix=inputs=2:duration=shortest,volume=0.03,"
        f"afade=t=in:st=0:d=2,afade=t=out:st={max(0.0, duration_s - 2):.2f}:d=2",
        "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)
    music = sb["music"]
    duration_s = float(music["duration_s"])

    if CHOSEN.exists() and CHOSEN.stat().st_size > 1000:
        print(f"[music] using Kaggle ACE-Step output: {CHOSEN}")
        return

    key = sha("music-placeholder", duration_s, music.get("bpm"), music.get("seed"))

    def produce(path: Path) -> None:
        logged_call(
            stage="music",
            shot="score",
            model="ffmpeg-placeholder-drone",
            seed=music.get("seed"),
            params={"duration_s": duration_s},
            fn=lambda: generate_placeholder(duration_s, path),
        )

    cached("music", "placeholder", key, PLACEHOLDER, produce)
    print(f"[music] Note: {CHOSEN} not found; using {PLACEHOLDER} until ACE-Step output is placed there.")


if __name__ == "__main__":
    main()

