"""Print a one-line summary per run: progress, speed, and the latest/best eval F1.

    python scripts/summarize_runs.py [runs_dir] [--glob 'bench_*']

Uses only the standard library, so it runs with any python3 on any node.
"""

import argparse
import json
from pathlib import Path


def summarize(metrics_path: Path) -> str:
    train, evals = [], []
    for line in metrics_path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line being written right now
        (evals if d.get("event") == "eval" else train).append(d)
    run = metrics_path.parent.name
    if not train and not evals:
        return f"{run:40s} (no data yet)"
    last = train[-1] if train else {}
    speeds = [t["steps_per_sec"] for t in train[-5:] if "steps_per_sec" in t]
    best = max(evals, key=lambda e: e.get("eval/f1", 0)) if evals else {}
    parts = [
        f"{run:40s}",
        f"step {last.get('step', 0):>10,}",
        f"{(sum(speeds) / len(speeds)) if speeds else 0:7.0f} steps/s",
        f"{last.get('hours', 0):5.2f} h",
        f"rollout F1 {last.get('rollout/f1', float('nan')):.3f}",
    ]
    if evals:
        parts += [
            f"eval F1 {evals[-1].get('eval/f1', float('nan')):.3f} @ {evals[-1]['step']:,}",
            f"best {best.get('eval/f1', float('nan')):.3f} @ {best.get('step', 0):,}",
        ]
        if "eval/note_f1" in evals[-1]:  # runs started after note metrics were added
            best_note = max(evals, key=lambda e: e.get("eval/note_f1", 0))
            parts.append(f"note F1 {evals[-1]['eval/note_f1']:.3f} (best {best_note['eval/note_f1']:.3f}), "
                         f"presses {int(evals[-1].get('eval/presses', 0))}/{int(evals[-1].get('eval/strikes', 0))}")
    return "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--glob", default="*")
    args = p.parse_args()
    for m in sorted(Path(args.runs_dir).glob(f"{args.glob}/metrics.jsonl")):
        print(summarize(m))


if __name__ == "__main__":
    main()
