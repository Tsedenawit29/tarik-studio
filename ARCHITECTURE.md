# ARCHITECTURE

## 1. Goal and design principle

Turn one file, `storyboard.json`, into a finished 60–90 s film (9:16 and 16:9) with a single command.
Each stage reads the storyboard, writes into `outputs/<stage>/`, and is cached by a hash of its inputs, so
changing one shot regenerates only that shot. Models are replaceable; the contract and the logging are not.

## 2. Data flow

```
                        storyboard.json  (the contract)
                               │
                    schema/storyboard.schema.json
                               │
                         [0] validate ──► consent gate (ETHICS.md)
                               │
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
 [1] narration          [2] keyframes            [4] music
 Gemini TTS             FLUX (Cloudflare)        ACE-Step 1.5 (Kaggle)
 1 wav per shot         ref sheet → 1 png/shot   3 candidates → 1 chosen
        │                      │                       │
        │                      ▼                       │
        │               [3] shots                      │
        │               Wan 2.2 (Kaggle) or            │
        │               Ken Burns fallback             │
        │               1 mp4 per shot                 │
        └──────────────┬───────┴───────────────────────┘
                       ▼
                 [5] assemble  (FFmpeg, deterministic)
        timing from narration → cut → subtitles → duck → loudness
                       ▼
        film_9x16.mp4   film_16x9.mp4   (+ AI end card)

 Every model call, in every stage ──► runs.jsonl
```

## 3. The contract: storyboard.json

Validated against `schema/storyboard.schema.json` before every run. Top level: title, language, audience,
intent, style_block, character, narration_style, music, shots[8–12], end_card.

Each shot holds everything needed to reproduce it: `id`, `reason` (why it exists), `shot_size`,
`image_prompt`, `motion_prompt` (motion and camera only), `motion` (`wan` or `kenburns`), `seed`, and
`narration` (`text` in the film language, `english` for subtitles).

Prompts and seeds live in the storyboard, never in code, so a run is reproducible and a change is a diff.

## 4. Stages

| # | Stage | Input (cache key = hash of) | Output | Model / tool | Where it runs |
|---|---|---|---|---|---|
| 0 | validate | storyboard.json, schema | pass/fail | jsonschema | local |
| 1 | narration | shot.narration.text, narration_style, voice, model | `outputs/narration/sNN.wav` | Gemini TTS | Gemini API |
| 2a | reference | style_block, character, seed | `outputs/keyframes/ref_*.png` | FLUX | Cloudflare |
| 2b | keyframes | image_prompt, style_block, character, seed, model | `outputs/keyframes/sNN.png` | FLUX | Cloudflare |
| 3 | shots | keyframe file hash, motion_prompt, seed, motion type | `outputs/shots/sNN.mp4` | Wan 2.2 TI2V-5B, else FFmpeg zoompan | Kaggle / local |
| 4 | music | caption, bpm, key, duration, seed | `outputs/music/cand_*.wav`, `chosen.wav` | ACE-Step 1.5 | Kaggle |
| 5 | assemble | all of the above, shot durations, end card | `outputs/final/film_*.mp4`, `.srt` | FFmpeg | local |

Stages 1, 2 and 4 are independent and can run in parallel. Stage 3 depends on 2. Stage 5 depends on all.

## 5. Timing: narration drives the picture

Shot duration = max(narration length + 0.6 s breathing room, `min_duration_s`). The assemble stage measures
each narration wav with ffprobe, so the radio cut, the rough cut and the final cut all share one timeline.
A Wan clip is about 5 s; if the shot is longer, the last frame is held or the clip is slowed slightly, and
if it is shorter, the clip is trimmed. Narration is generated line by line so a single re-take is cheap.

## 6. Caching

`common.cached()` stores a `<output>.key` file beside each output containing the hash of its inputs. If the
key matches and the file exists, the stage skips it. Editing shot s05's image prompt changes only
s05's keyframe key, which changes s05's clip key, which changes the final cut. Other shots are untouched.
This is what the walkthrough demonstrates live.

## 7. Logging: runs.jsonl

One JSON line per model call, written by `common.logged_call()`, including failures:
`ts, stage, shot, model, params, seed, latency_s, cost_usd_list_price, status, error`.
Notebook stages (Wan, ACE-Step) append the same fields with a `gpu_minutes` value, and the file is merged
back into the repo. COSTS.md is computed from this file, not written by hand.

## 8. Failure handling

- **Retries:** 3 attempts with exponential backoff on every call (`common.retry`).
- **Quota exhausted:** stages stop cleanly and name the shot; cached work is kept, and the next run resumes.
- **Three-attempt rule:** after 3 failed generations of a shot, fall back (Wan → Ken Burns) and log why.
- **GPU sessions time out:** notebooks save each output the moment it finishes and skip anything already on disk.
- **Bad keyframe:** a quick manual check (faces, hands) before spending GPU time animating it.
- **Secrets:** only from `.env` (gitignored); `.env.example` is committed.
- **Consent gate:** `validate` refuses to run if `character.real_person` is true and ETHICS.md lacks `CONSENT: GRANTED`.

## 9. Consistency strategy

1. Generate a character and style reference sheet first, with several options, and pick one (golden reference).
2. Attach the same `style_block` and `character.description` to every image prompt, verbatim.
3. Animate from keyframes (image-to-video), not text-to-video.
4. Motion prompts describe only motion and camera, never the subject again.
5. Keep the subject centred so both aspect ratios work from the same material.

## 10. Assembly details (FFmpeg)

- **Aspect ratios:** shots are generated at 16:9 with the subject centred; the 9:16 export is a centre crop
  (then scaled), so one set of shots serves both outputs.
- **Subtitles:** an SRT is built from the narration timings and the English lines, at most 2 lines of about
  40 characters each, burned in and kept inside the vertical safe area.
- **Mix:** narration and music are each brought to similar loudness first, then the music is ducked with
  `sidechaincompress` keyed by the narration (start at threshold 0.02, ratio 6, attack 300 ms,
  release 1500 ms, tune by ear).
- **Loudness:** `loudnorm` two-pass to about -14 LUFS, true peak near -1 dB.
- **End card:** 2–3 s "Made with AI" card appended in code.
- **Encode:** H.264 + AAC MP4, constant frame rate (24 fps).

## 11. Repository layout

```
storyboard.json      schema/storyboard.schema.json
pipeline/            common.py validate.py smoke_test.py narration.py keyframes.py shots.py music.py assemble.py
notebooks/           wan_shots.ipynb  acestep_music.ipynb      (Kaggle)
outputs/             narration/ keyframes/ shots/ music/ final/
runs.jsonl           DESIGN.md EXPERIMENTS.md FAILURES.md COSTS.md ETHICS.md
Makefile             make film | validate | narration | keyframes | shots | music | assemble | smoke
```

## 12. Key decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| One JSON contract | Stages stay independent; one shot can be regenerated alone | Schema must be kept in sync |
| Narration sets timing | Matches professional practice and makes the radio cut a real prototype | Retiming needs narration regenerated |
| Wan on Kaggle, not in `make film` | GPU is the scarce resource, and sessions are short | Clips are downloaded into `outputs/shots/` by hand |
| Ken Burns fallback | A narrated, scored sequence of stills is still a film | Less motion |
| 16:9 generation, centre-crop to 9:16 | Halves generation cost | Wide scenes lose side content |
| Local FFmpeg assembly | Deterministic and free | Needs FFmpeg with libass installed |

## 13. Scope tiers (what ships if time runs out)

1. **Floor:** narration + Ken Burns stills + score + subtitles + end card, both ratios.
2. **Target:** Wan 2.2 clips on the 3–4 strongest shots, Ken Burns elsewhere.
3. **Ceiling:** extra language, a blind preference test, or a benchmark.