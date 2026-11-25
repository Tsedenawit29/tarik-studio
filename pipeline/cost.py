"""Summarize runs.jsonl by stage and model (calls, failures, latency, list-price cost)."""
from __future__ import annotations

import json
from collections import defaultdict

from common import RUNS_LOG


def main() -> None:
    if not RUNS_LOG.exists() or not RUNS_LOG.read_text(encoding="utf-8").strip():
        print("No entries in runs.jsonl yet.")
        return

    stats = defaultdict(lambda: {"ok": 0, "error": 0, "latency_s": 0.0, "cost_usd": 0.0})
    for line in RUNS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        k = (row.get("stage", "?"), row.get("model", "?"))
        if row.get("status") == "ok":
            stats[k]["ok"] += 1
        else:
            stats[k]["error"] += 1
        stats[k]["latency_s"] += float(row.get("latency_s") or 0.0)
        stats[k]["cost_usd"] += float(row.get("cost_usd_list_price") or 0.0)

    print("| Stage | Model | OK | Errors | Total Latency (s) | List-Price (USD) |")
    print("|---|---|---:|---:|---:|---:|")
    for (stage, model), s in sorted(stats.items()):
        print(f"| {stage} | {model} | {s['ok']} | {s['error']} | {s['latency_s']:.2f} | ${s['cost_usd']:.4f} |")


if __name__ == "__main__":
    main()

