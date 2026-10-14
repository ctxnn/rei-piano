"""Build the training dashboard's data folder from training runs.

    # live, on a training node: rebuild every 30 s, serve on port 8000
    python -m runaway.dashboard.build --runs runs --out dashboard_site --watch --serve

    # once, for the final replayable export
    python -m runaway.dashboard.build --runs runs --out dashboard_site

The page (dashboard/app.html) is static. It only reads files this script writes:

    <out>/index.html                       the page
    <out>/data/index.json                  all runs: status, progress, best F1, settings
    <out>/data/runs/<run>/metrics.jsonl    copied as-is from the run (all curves)
    <out>/data/runs/<run>/evals.json       one entry per evaluation (step, F1, files)
    <out>/data/runs/<run>/rolls.json       piano rolls (target vs played) of every eval,
                                           keyed by "step_N" (one file per run keeps the
                                           published page under the file-count limit;
                                           rolls/step_N.json are the per-eval build cache)
    <out>/data/runs/<run>/videos/step_N.mp4   the performance, with sound

Videos are rendered here from the recorded simulator states (see runaway/render.py).
On a node without GPU graphics we render with OSMesa (software OpenGL): set
MUJOCO_GL=osmesa. A 60 s performance at 480x270 takes ~30-60 s of one CPU core, so the
renderer runs at low priority next to training and only renders what is new.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from runaway.evaluate import note_metrics

APP_HTML = Path(__file__).with_name("app.html")

# The page is authored as a body fragment (that is the format claude.ai artifacts
# expect). For serving it directly we wrap it in a minimal document.
_SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{{color-scheme:light}}body{{margin:0;font:14px system-ui,sans-serif}}
img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head><body>
{body}
</body></html>
"""


# ----------------------------------------------------------------------------- data


def read_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now
    return rows


def piano_roll(recording: dict) -> dict:
    """Compress a recording into note intervals for the page's piano roll.

    For each key that was either wanted or pressed, list [start, end) intervals in
    control steps, separately for "target" (the MIDI) and "played" (the robot).
    """
    dt = float(recording["control_timestep"])
    goal = recording["goal"].astype(bool)
    played = recording["key_activation"].astype(bool)

    def intervals(column: np.ndarray) -> list[list[int]]:
        padded = np.concatenate([[False], column, [False]]).astype(np.int8)
        d = np.diff(padded)
        return [[int(s), int(e)] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    keys = sorted(set(np.flatnonzero(goal.any(0))) | set(np.flatnonzero(played.any(0))))
    return {
        "dt": dt,
        "steps": int(goal.shape[0]),
        "keys": [
            {"midi": int(k) + 21, "target": intervals(goal[:, k]), "played": intervals(played[:, k])}
            for k in keys
        ],
    }


def run_status(metrics: list[dict], total_steps: int, metrics_path: Path) -> str:
    if not metrics:
        return "starting"
    last_step = max(m.get("step", 0) for m in metrics)
    if last_step >= total_steps:
        return "finished"
    stale = time.time() - metrics_path.stat().st_mtime > 600
    return "stopped early" if stale else "training"


def pick_video_steps(evals: list[tuple[int, float]], limit: int | None) -> set[int]:
    """Which evaluations get a video: all, or `limit` of them (first, best, last, and
    evenly spaced ones in between) to keep a published dashboard small."""
    steps = [s for s, _ in evals]
    if limit is None or len(steps) <= limit:
        return set(steps)
    chosen = {steps[0], steps[-1], max(evals, key=lambda e: e[1])[0]}
    for i in np.linspace(0, len(steps) - 1, limit).round().astype(int):
        if len(chosen) >= limit:
            break
        chosen.add(steps[i])
    return chosen


def build_run(run_dir: Path, out_dir: Path, render_videos: bool, video_size: tuple[int, int],
              max_videos: int | None = None) -> dict:  # fmt: skip
    cfg = json.loads((run_dir / "config.json").read_text())
    metrics_path = run_dir / "metrics.jsonl"
    metrics = read_metrics(metrics_path) if metrics_path.exists() else []
    dst = out_dir / "data" / "runs" / run_dir.name
    (dst / "rolls").mkdir(parents=True, exist_ok=True)
    (dst / "videos").mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        shutil.copyfile(metrics_path, dst / "metrics.jsonl")

    npzs = sorted((run_dir / "evals").glob("step_*.npz"))
    scores = []
    for npz in npzs:
        try:
            with np.load(npz) as rec:
                scores.append((int(npz.stem.split("_")[1]), float(rec["metric_f1"])))
        except Exception:
            scores.append((int(npz.stem.split("_")[1]), 0.0))
    video_steps = pick_video_steps(scores, max_videos)

    evals = []
    for npz in npzs:
        step = int(npz.stem.split("_")[1])
        roll_path = dst / "rolls" / f"{npz.stem}.json"
        video_path = dst / "videos" / f"{npz.stem}.mp4"
        try:
            if not roll_path.exists():
                with np.load(npz) as rec:
                    roll_path.write_text(json.dumps(piano_roll(dict(rec))))
            if render_videos and step in video_steps and not video_path.exists():
                _render_video(npz, video_path, cfg, video_size)
        except Exception as exc:  # a half-written .npz, or a render failure
            print(f"[dashboard] skipping {npz}: {exc}", flush=True)
        with np.load(npz) as rec:
            m = {k[len("metric_"):]: float(rec[k]) for k in rec.files if k.startswith("metric_")}
            if "note_f1" not in m:  # recordings made before note metrics existed
                m.update(note_metrics(rec["goal"], rec["key_activation"]))
        evals.append({
            "step": step,
            **m,
            "roll": npz.stem if roll_path.exists() else None,
            "video": f"videos/{npz.stem}.mp4" if video_path.exists() else None,
        })  # fmt: skip
    # Evaluations whose recording is not on this machine (only the light metrics were
    # pulled) still have their scores in metrics.jsonl: list them without roll or video.
    have = {e["step"] for e in evals}
    logged = {m["step"]: m for m in metrics if m.get("event") == "eval" and m["step"] not in have}
    for step, m in logged.items():
        evals.append({"step": step, **{k[len("eval/"):]: v for k, v in m.items() if k.startswith("eval/")},
                      "roll": None, "video": None})  # fmt: skip
    evals.sort(key=lambda e: e["step"])
    (dst / "evals.json").write_text(json.dumps(evals))
    bundle = {e["roll"]: json.loads((dst / "rolls" / f"{e['roll']}.json").read_text())
              for e in evals if e["roll"]}  # fmt: skip
    (dst / "rolls.json").write_text(json.dumps(bundle))

    train = [m for m in metrics if m.get("event") == "train"]
    best = max(evals, key=lambda e: e.get("f1", 0), default=None)
    best_note = max(evals, key=lambda e: e.get("note_f1", 0), default=None)
    return {
        "name": run_dir.name,
        "status": run_status(metrics, cfg["total_env_steps"], metrics_path) if metrics_path.exists() else "starting",
        "step": train[-1]["step"] if train else 0,
        "total_steps": cfg["total_env_steps"],
        "hours": train[-1].get("hours", 0) if train else 0,
        "steps_per_sec": train[-1].get("steps_per_sec") if train else None,
        "best_f1": best.get("f1") if best else None,
        "best_step": best["step"] if best else None,
        "best_note_f1": best_note.get("note_f1") if best_note else None,
        "best_note_step": best_note["step"] if best_note else None,
        "last_f1": evals[-1].get("f1") if evals else None,
        "num_evals": len(evals),
        "settings": {
            "embodiment": cfg["env"].get("embodiment", "hands"),
            "num_envs": cfg["num_envs"],
            "discount": cfg["sac"]["discount"],
            "batch_size": cfg["sac"]["batch_size"],
            "updates_per_env_step": cfg["updates_per_env_step"],
            "seed": cfg["seed"],
            "midi": Path(cfg["env"]["midi_path"]).name,
            "lookahead": cfg["env"]["n_steps_lookahead"],
            # G1 variants (defaults = what older runs used before the option existed).
            "arm_control": cfg["env"].get("arm_control", "joint"),
            "piano_height": cfg["env"].get("piano_height", 0.75),
            "extra_press_penalty": cfg["env"].get("extra_press_penalty", 0.0),
            "arm_action_smoothing": cfg["env"].get("arm_action_smoothing", 0.0),
            "arm_action_rate_penalty": cfg["env"].get("arm_action_rate_penalty", 0.0),
            "warm_start": bool(cfg.get("init_from")),
        },
    }


def _render_video(npz: Path, video_path: Path, cfg: dict, size: tuple[int, int]) -> None:
    from runaway.render import RenderArgs, render

    tmp = video_path.with_suffix(".tmp.mp4")
    embodiment = cfg["env"].get("embodiment", "hands")
    camera = "humanoid_front_left" if embodiment == "g1" else "piano/back"
    render(RenderArgs(recording=npz, out=tmp, width=size[0], height=size[1], camera=camera,
                      midi_path=cfg["env"]["midi_path"], embodiment=embodiment,
                      lead_in=cfg["env"]["initial_buffer_time"], substeps=1, crf=30, audio_kbps=96))  # fmt: skip
    tmp.rename(video_path)  # atomic: the page never sees a half-written video


def build(runs_dir: Path, out_dir: Path, render_videos: bool, video_size, live: bool,
          only: str | None = None, max_videos: int | None = None) -> None:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(_SKELETON.format(body=APP_HTML.read_text()))
    (out_dir / "app.html").write_text(APP_HTML.read_text())  # fragment, for artifact publishing
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "config.json").exists()):
        if only and not run_dir.match(only):
            continue
        runs.append(build_run(run_dir, out_dir, render_videos, video_size, max_videos))
    index = {
        "project": "Runaway — robot hands learning Kanye West's Runaway",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live": live,
        "runs": runs,
    }
    (out_dir / "data").mkdir(exist_ok=True)
    tmp = out_dir / "data" / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1))
    tmp.rename(out_dir / "data" / "index.json")


def serve(out_dir: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[dashboard] serving {out_dir} on :{port}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("dashboard_site"))
    p.add_argument("--only", help="glob of run names to include, e.g. 'r1_*'")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--video-size", default="480x270")
    p.add_argument("--max-videos-per-run", type=int, help="render only this many evals per run")
    p.add_argument("--watch", action="store_true", help="rebuild every --interval seconds")
    p.add_argument("--interval", type=float, default=30)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    size = tuple(int(x) for x in args.video_size.split("x"))
    if args.serve:
        serve(args.out, args.port)
    while True:
        t0 = time.time()
        build(args.runs, args.out, not args.no_videos, size, live=args.watch, only=args.only,
              max_videos=args.max_videos_per_run)
        print(f"[dashboard] built in {time.time() - t0:.1f}s", flush=True)
        if not args.watch:
            break
        time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    os.nice(10) if hasattr(os, "nice") else None  # stay out of training's way
    main()
