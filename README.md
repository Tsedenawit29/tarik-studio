# tarik-studio

A small studio in code that turns a written storyboard into a 60–90 s narrated, scored, subtitled short film
(9:16 and 16:9). Built for the *Heritage Short* challenge. `storyboard.json` is the single contract between
stages; change one shot and only that shot regenerates.

> Film: _link_ · Walkthrough: _link_ · Post: _link_

## Run it

```bash
git clone <this-repo> && cd tarik-studio
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # also needs ffmpeg on PATH
cp .env.example .env                     # add your keys
make film                                # validate → narration → keyframes → shots → music → assemble
```

Outputs land in `outputs/`. Re-running `make film` skips every stage whose inputs hash is unchanged.
Individual stages: `make validate | narration | keyframes | shots | music | assemble | smoke`.

## Pipeline

| Stage | Script | Model | Runs on |
|---|---|---|---|
| validate | `pipeline/validate.py` | JSON Schema | local |
| narration | `pipeline/narration.py` | Gemini TTS | Gemini API |
| keyframes | `pipeline/keyframes.py` | FLUX (Cloudflare) | Workers AI |
| shots | `pipeline/shots.py` | Wan 2.2 TI2V-5B (notebook) / Ken Burns fallback | Kaggle + FFmpeg |
| music | `pipeline/music.py` | ACE-Step 1.5 (notebook) | Kaggle |
| assemble | `pipeline/assemble.py` | FFmpeg | local |

Wan 2.2 and ACE-Step run in the notebooks under `notebooks/`; their outputs are downloaded into `outputs/`
and picked up by the stages above. See ARCHITECTURE.md.

## Docs

DESIGN.md · ARCHITECTURE.md · EXPERIMENTS.md · FAILURES.md · COSTS.md · ETHICS.md · `runs.jsonl` (one line per model call)

## Cost

Everything runs on free tiers; measured numbers and list-price equivalents are in COSTS.md.
