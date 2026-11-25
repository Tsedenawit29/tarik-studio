"""Stage 5: Assemble final 16:9 and 9:16 films (timing, xfade, end card, subtitles, ducking, loudnorm)."""
from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path

from common import OUTPUTS, load_env, load_storyboard, logged_call

NARRATION_DIR = OUTPUTS / "narration"
SHOTS_DIR = OUTPUTS / "shots"
MUSIC_DIR = OUTPUTS / "music"
FINAL_DIR = OUTPUTS / "final"

XFADE_S = 0.4
FPS = 24


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(res.stdout)["format"]["duration"])


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap_subtitle(text: str, width: int = 40) -> str:
    lines = textwrap.wrap(text.strip(), width=width)
    if len(lines) > 2:
        # Re-wrap into 2 balanced lines if needed
        mid = len(text) // 2
        split_at = text.rfind(" ", 0, mid + 10)
        if split_at != -1:
            lines = [text[:split_at].strip(), text[split_at:].strip()]
        else:
            lines = lines[:2]
    return "\n".join(lines)


def build_timeline(sb: dict) -> tuple[list[dict], float]:
    """Compute exact start/end timestamps for every shot and narration."""
    timeline = []
    cursor = 0.0
    for i, shot in enumerate(sb["shots"]):
        sid = shot["id"]
        clip = SHOTS_DIR / f"{sid}.mp4"
        wav = NARRATION_DIR / f"{sid}.wav"
        if not clip.exists():
            raise SystemExit(f"Missing shot clip {clip}. Run `make shots` first.")
        clip_dur = probe_duration(clip)
        wav_dur = probe_duration(wav) if wav.exists() else 0.0
        timeline.append({
            "id": sid,
            "clip": clip,
            "wav": wav if wav.exists() else None,
            "start": cursor,
            "duration": clip_dur,
            "wav_start": cursor + (0.25 if i > 0 else 0.2),
            "wav_duration": wav_dur,
            "english": shot["narration"]["english"],
        })
        cursor += clip_dur - XFADE_S
    total_shots_dur = cursor + XFADE_S  # last shot has no trailing xfade overlap yet
    return timeline, total_shots_dur


def write_srt(timeline: list[dict], srt_path: Path) -> None:
    blocks = []
    for idx, item in enumerate(timeline, start=1):
        start = item["wav_start"]
        dur = item["wav_duration"] if item["wav_duration"] > 0 else max(2.0, item["duration"] - 0.6)
        end = min(start + dur + 0.2, item["start"] + item["duration"] - 0.1)
        blocks.append(
            f"{idx}\n{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n{_wrap_subtitle(item['english'])}\n"
        )
    srt_path.write_text("\n".join(blocks), encoding="utf-8")


def render_end_card(text: str, duration_s: float, out_path: Path) -> None:
    safe_text = text.replace(":", "\\:").replace("'", "\\'")
    vf = (
        f"drawtext=text='{safe_text}':fontcolor=white@0.9:fontsize=36:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:alpha='if(lt(t,0.5),t/0.5,if(gt(t,{duration_s - 0.5:.2f}),({duration_s:.2f}-t)/0.5,1))',"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=#0d1117:s=1280x720:r={FPS}:d={duration_s:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", str(FPS), "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def concat_video_xfade(clips: list[Path], durations: list[float], out_path: Path) -> None:
    """Concatenate video clips with 0.4s crossfades."""
    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    filters = []
    last_label = "[0:v]"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1] - XFADE_S
        out_label = f"[v{i}]" if i < len(clips) - 1 else "[vout]"
        filters.append(
            f"{last_label}[{i}:v]xfade=transition=fade:duration={XFADE_S}:offset={offset:.3f}{out_label}"
        )
        last_label = out_label

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", str(FPS), "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def mix_audio(timeline: list[dict], music_path: Path, total_dur: float, out_wav: Path) -> None:
    """Mix narration tracks + sidechain-ducked music, then two-pass loudnorm to -14 LUFS."""
    narr_items = [t for t in timeline if t["wav"] is not None]
    raw_mix = FINAL_DIR / "_raw_mix.wav"

    if not narr_items:
        # Music-only fallback
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(music_path),
            "-t", f"{total_dur:.3f}",
            "-af", f"afade=t=out:st={max(0.0, total_dur - 2.0):.2f}:d=2.0",
            "-ar", "44100", "-ac", "2",
            str(raw_mix),
        ]
        subprocess.run(cmd, check=True)
    else:
        inputs = ["-i", str(music_path)]
        filters = []
        delayed_labels = []
        for idx, item in enumerate(narr_items, start=1):
            inputs.extend(["-i", str(item["wav"])])
            delay_ms = int(round(item["wav_start"] * 1000))
            lbl = f"[n{idx}]"
            filters.append(
                f"[{idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"adelay={delay_ms}|{delay_ms},loudnorm=I=-16:TP=-1.5:LRA=11{lbl}"
            )
            delayed_labels.append(lbl)

        filters.append(
            f"{''.join(delayed_labels)}amix=inputs={len(delayed_labels)}:duration=longest:normalize=0,asplit=2[narr_sc][narr_mix]"
        )
        filters.append(
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,"
            f"aloop=loop=-1:size=2e+09,atrim=0:{total_dur:.3f},"
            f"afade=t=out:st={max(0.0, total_dur - 2.5):.2f}:d=2.5,"
            f"loudnorm=I=-22:TP=-2.0:LRA=11[mus_norm]"
        )
        filters.append(
            "[mus_norm][narr_sc]sidechaincompress=threshold=0.02:ratio=6:attack=300:release=1500[mus_ducked]"
        )
        filters.append(
            "[narr_mix][mus_ducked]amix=inputs=2:duration=longest:normalize=0[aout]"
        )

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            *inputs,
            "-filter_complex", ";".join(filters),
            "-map", "[aout]",
            "-t", f"{total_dur:.3f}",
            "-ar", "44100", "-ac", "2",
            str(raw_mix),
        ]
        subprocess.run(cmd, check=True)

    # Two-pass loudnorm to -14 LUFS, -1 dB TP
    pass1 = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_mix),
            "-af", "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=True,
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", pass1.stderr, re.DOTALL)
    if m:
        stats = json.loads(m.group(0))
        ln_filter = (
            f"loudnorm=I=-14:TP=-1.0:LRA=11:linear=true:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}"
        )
    else:
        ln_filter = "loudnorm=I=-14:TP=-1.0:LRA=11"

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_mix),
            "-af", ln_filter,
            "-ar", "44100", "-ac", "2",
            str(out_wav),
        ],
        check=True,
    )
    raw_mix.unlink(missing_ok=True)


def export_final(video_16x9: Path, audio_wav: Path, srt_path: Path, out_16x9: Path, out_9x16: Path) -> None:
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")

    # 16:9 with burned-in subtitles
    vf_16x9 = (
        f"subtitles='{srt_escaped}':force_style='FontName=DejaVu Sans,FontSize=20,"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=28'"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_16x9), "-i", str(audio_wav),
            "-vf", vf_16x9,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-r", str(FPS), "-shortest",
            str(out_16x9),
        ],
        check=True,
    )

    # 9:16 centre-crop + scale + burned-in subtitles in safe vertical area
    vf_9x16 = (
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=720:1280:flags=lanczos,"
        f"subtitles='{srt_escaped}':force_style='FontName=DejaVu Sans,FontSize=16,"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=90'"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_16x9), "-i", str(audio_wav),
            "-vf", vf_9x16,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-r", str(FPS), "-shortest",
            str(out_9x16),
        ],
        check=True,
    )


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    music_path = MUSIC_DIR / "chosen.wav"
    if not music_path.exists():
        music_path = MUSIC_DIR / "placeholder.wav"
    if not music_path.exists():
        raise SystemExit("No music found. Run `make music` first.")

    timeline, shots_dur = build_timeline(sb)
    end_card_dur = float(sb["end_card"]["duration_s"])
    total_dur = shots_dur + end_card_dur - XFADE_S

    srt_path = FINAL_DIR / "subtitles.srt"
    write_srt(timeline, srt_path)

    end_card_mp4 = FINAL_DIR / "_end_card.mp4"
    render_end_card(sb["end_card"]["text"], end_card_dur, end_card_mp4)

    all_clips = [t["clip"] for t in timeline] + [end_card_mp4]
    all_durs = [t["duration"] for t in timeline] + [end_card_dur]

    video_16x9 = FINAL_DIR / "_video_16x9.mp4"
    audio_wav = FINAL_DIR / "_mix.wav"
    out_16x9 = FINAL_DIR / "film_16x9.mp4"
    out_9x16 = FINAL_DIR / "film_9x16.mp4"

    def _run() -> None:
        concat_video_xfade(all_clips, all_durs, video_16x9)
        mix_audio(timeline, music_path, total_dur, audio_wav)
        export_final(video_16x9, audio_wav, srt_path, out_16x9, out_9x16)

    logged_call(
        stage="assemble",
        shot="final",
        model="ffmpeg",
        params={"shots": len(timeline), "total_s": round(total_dur, 2)},
        fn=_run,
    )

    for tmp in (end_card_mp4, video_16x9, audio_wav):
        tmp.unlink(missing_ok=True)

    print(f"[assemble] wrote {out_16x9} and {out_9x16} ({total_dur:.1f}s)")


if __name__ == "__main__":
    main()

