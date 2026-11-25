"""Shared helpers: storyboard loading + validation, content-hash caching, retries, run logging.

Every model call in every stage must go through `logged_call` so runs.jsonl stays complete.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
STORYBOARD = ROOT / "storyboard.json"
SCHEMA = ROOT / "schema" / "storyboard.schema.json"
OUTPUTS = ROOT / "outputs"
RUNS_LOG = ROOT / "runs.jsonl"


def load_env() -> None:
    """Tiny .env loader (no dependency). Secrets live in .env, which is gitignored."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_storyboard(validate: bool = True) -> dict:
    sb = json.loads(STORYBOARD.read_text(encoding="utf-8"))
    if validate:
        validate_storyboard(sb)
    return sb


def validate_storyboard(sb: dict) -> None:
    import jsonschema  # pip install jsonschema

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(sb), key=lambda e: list(e.path))
    if errors:
        msg = "\n".join(f"  - {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors)
        raise SystemExit(f"storyboard.json is invalid:\n{msg}")
    ids = [s["id"] for s in sb["shots"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("storyboard.json is invalid: duplicate shot ids")


def sha(*parts: Any, length: int = 16) -> str:
    """Stable content hash of JSON-serialisable inputs plus any file contents given as Path."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            h.update(p.read_bytes() if p.exists() else b"<missing>")
        else:
            h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:length]


def log_run(**fields: Any) -> None:
    fields.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with RUNS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fields, ensure_ascii=False) + "\n")


def retry(attempts: int = 3, base_delay: float = 2.0) -> Callable:
    """Exponential backoff. Failures are re-raised after the last attempt."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for i in range(attempts):
                try:
                    return fn(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    if i == attempts - 1:
                        raise
                    delay = base_delay * 2**i
                    print(f"  retry {i + 1}/{attempts - 1} after {type(e).__name__}: {e} (sleep {delay:.0f}s)")
                    time.sleep(delay)

        return wrapper

    return deco


def logged_call(stage: str, model: str, fn: Callable[[], Any], *, params: dict | None = None,
                seed: int | None = None, cost_usd_list_price: float = 0.0, shot: str | None = None) -> Any:
    """Run fn(), time it, and append one line to runs.jsonl (success or failure)."""
    t0 = time.perf_counter()
    try:
        result = retry()(fn)()
        status, err = "ok", None
        return result
    except Exception as e:  # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        log_run(stage=stage, shot=shot, model=model, params=params or {}, seed=seed,
                latency_s=round(time.perf_counter() - t0, 3), cost_usd_list_price=cost_usd_list_price,
                status=status, error=err)


def cached(stage: str, shot: str, key: str, out: Path, produce: Callable[[Path], None]) -> bool:
    """Regenerate `out` only if the input hash `key` changed. Returns True if it regenerated."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = out.with_suffix(out.suffix + ".key")
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        print(f"[{stage}] {shot}: cached ({key})")
        return False
    print(f"[{stage}] {shot}: generating ({key})")
    produce(out)
    stamp.write_text(key)
    return True
