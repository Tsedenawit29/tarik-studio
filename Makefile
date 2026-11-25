PY ?= python3

.PHONY: film validate narration keyframes shots music assemble smoke plan cost clean

# One command: regenerate the film from storyboard.json (cached stages are skipped).
film: validate narration keyframes shots music assemble

validate:
	$(PY) pipeline/validate.py

plan:
	$(PY) pipeline/plan.py

cost:
	$(PY) pipeline/cost.py

smoke:
	$(PY) pipeline/smoke_test.py

narration: validate
	$(PY) pipeline/narration.py

keyframes: validate
	$(PY) pipeline/keyframes.py

# Wan 2.2 runs on Kaggle; this stage picks up clips you downloaded into outputs/shots/
# and falls back to Ken Burns on the keyframe for any shot without a clip.
shots: keyframes
	$(PY) pipeline/shots.py

music: validate
	$(PY) pipeline/music.py

assemble: narration shots music
	$(PY) pipeline/assemble.py

clean:
	rm -rf outputs/*
