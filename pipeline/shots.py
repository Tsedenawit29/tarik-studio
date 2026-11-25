"""Stage 3: Produce per-shot 16:9 24fps MP4s from Wan 2.2 clips or Ken Burns (FFmpeg zoompan) fallback."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common import OUTPUTS, cached, load_env, load_storyboard, logged_call, sha

SHOTS_DIR = OUTPUTS / "shots"
WAN_DIR = SHOTS_DIR / "wan"
KEYFRAMES_DIR = OUTPUTS / "keyframes"
NARRATION_DIR = OUTPUTS / "narration"

FPS = 24
OUT_W = 1280
OUT_H = 720


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(res.stdout)["format"]["duration"])


def target_duration(shot: dict) -> float:
    """Shot duration = max(narration_wav + 0.6s, min_duration_s)."""
    min_d = float(shot.get("min_duration_s", 5.0))
    wav = NARRATION_DIR / f"{shot['id']}.wav"
    if wav.exists():
        return round(max(probe_duration(wav) + 0.6, min_d), 3)
    return round(min_d, 3)


def _zoompan_filter(mode: str, frames: int) -> str:
    """Return FFmpeg filtergraph for smooth upscaled Ken Burns motion."""
    d = max(frames, 1)
    if mode == "push_in":
        z = f"1+0.15*on/{d}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "pull_out":
        z = f"1.15-0.15*on/{d}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "pan_left":
        z = "1.12"
        x = f"(iw-iw/zoom)*(1-on/{d})"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "pan_right":
        z = "1.12"
        x = f"(iw-iw/zoom)*(on/{d})"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "tilt_up":
        z = "1.12"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-on/{d})"
    else:
        z = f"1+0.12*on/{d}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    # Upscale 2x before zoompan to prevent sub-pixel jitter on slow moves
    return (
        f"scale={OUT_W * 2}:{OUT_H * 2}:flags=lanczos,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={d}:s={OUT_W}x{OUT_H}:fps={FPS},"
        f"format=yuv420p"
    )


def render_kenburns(keyframe: Path, mode: str, duration_s: float, out_path: Path) -> None:
    frames = int(round(duration_s * FPS))
    vf = _zoompan_filter(mode, frames)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-i", str(keyframe),
        "-vf", vf,
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", str(FPS), "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def conform_wan_clip(wan_clip: Path, duration_s: float, out_path: Path) -> None:
    """Conform a Kaggle Wan 2.2 MP4 to 1280x720 24fps and exact shot duration."""
    clip_dur = probe_duration(wan_clip)
    vf_parts = [
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase",
        f"crop={OUT_W}:{OUT_H}",
        f"fps={FPS}",
    ]
    if clip_dur < duration_s:
        pad_s = duration_s - clip_dur
        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad_s:.3f}")
    vf_parts.append("format=yuv420p")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(wan_clip),
        "-vf", ",".join(vf_parts),
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", str(FPS), "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)

    for shot in sb["shots"]:
        sid = shot["id"]
        kf = KEYFRAMES_DIR / f"{sid}.png"
        if not kf.exists():
            raise SystemExit(f"Missing keyframe {kf}. Run `make keyframes` first.")

        dur = target_duration(shot)
        wan_clip = WAN_DIR / f"{sid}.mp4"
        use_wan = shot.get("motion", "kenburns") == "wan" and wan_clip.exists()
        kb_mode = shot.get("kenburns", "push_in")

        if use_wan:
            key = sha("shot-wan", sid, wan_clip, dur)
            model_name = "wan-2.2-ti2v-5b-conform"
            params = {"source": "wan", "duration_s": dur}
            fn = lambda p, _w=wan_clip, _d=dur: conform_wan_clip(_w, _d, p)
        else:
            key = sha("shot-kb", sid, kf, kb_mode, dur)
            model_name = "ffmpeg-zoompan"
            params = {"source": "kenburns", "mode": kb_mode, "duration_s": dur}
            fn = lambda p, _k=kf, _m=kb_mode, _d=dur: render_kenburns(_k, _m, _d, p)

        out = SHOTS_DIR / f"{sid}.mp4"

        def produce(path: Path, _sid=sid, _model=model_name, _params=params, _fn=fn, _seed=shot["seed"]) -> None:
            logged_call(
                stage="shots",
                shot=_sid,
                model=_model,
                seed=_seed,
                params=_params,
                fn=lambda: _fn(path),
            )

        cached("shots", sid, key, out, produce)


if __name__ == "__main__":
    main()

