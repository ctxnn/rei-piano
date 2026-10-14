"""Evaluate a policy: play the song once, score it, and record it for video.

We record the *simulator state* (MuJoCo qpos/qvel) at every control step instead of
rendering frames during training. A performance is then a ~1 MB .npz file that can be
turned into a video later, on any machine, at any resolution (see render.py). This
keeps the training nodes free of OpenGL and keeps big video files off your laptop
until you actually want them.

Two kinds of metrics:

Frame metrics (RoboPianist's, per 50 ms control step, then averaged over the song):
  precision  of the keys the robot pressed, how many should have been pressed
  recall     of the keys that should have been pressed, how many the robot pressed
  f1         harmonic mean of the two; 1.0 = perfect performance

Note metrics (ours, `note_metrics`): how it *sounds*. Each strike in the song is a
"note onset" (a key going down). A robot onset counts as correct if it is on the right
key within +-100 ms of a target onset (each target matched at most once).
  note_recall     strikes played on time / strikes in the song
  note_precision  correct onsets / all onsets the robot made
  note_f1         harmonic mean
Why both: frame F1 barely notices a finger that chatters or brushes a neighbouring key
for 50 ms, but every such blip is an audible extra note. The G1 run with frame F1 0.876
made 253 key presses for 44 strikes: note F1 0.27 (docs/06_results.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import dm_env
import numpy as np


def _onsets(keys_down: np.ndarray) -> list[tuple[int, int]]:
    """(T, 88) booleans -> [(step, key)] for every key-down event."""
    padded = np.vstack([np.zeros((1, keys_down.shape[1]), bool), keys_down.astype(bool)])
    steps, keys = np.nonzero(np.diff(padded.astype(np.int8), axis=0) == 1)
    return list(zip(steps.tolist(), keys.tolist()))


def note_metrics(goal: np.ndarray, played: np.ndarray, tolerance_steps: int = 2) -> dict:
    """Onset-level precision/recall/F1 (see module docstring). 2 steps = 100 ms."""
    targets, presses = _onsets(goal), _onsets(played)
    free = {}  # key -> sorted onset steps not yet matched
    for step, key in presses:
        free.setdefault(key, []).append(step)
    hits = 0
    for step, key in targets:
        candidates = free.get(key, [])
        match = next((s for s in candidates if abs(s - step) <= tolerance_steps), None)
        if match is not None:
            candidates.remove(match)
            hits += 1
    precision = hits / len(presses) if presses else float(not targets)
    recall = hits / len(targets) if targets else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(note_precision=precision, note_recall=recall, note_f1=f1,
                strikes=len(targets), presses=len(presses))  # fmt: skip


def play_song(
    env: dm_env.Environment,
    policy: Callable[[np.ndarray], np.ndarray],
    record: bool = True,
) -> tuple[dict, dict | None]:
    """Run one full episode with `policy` (obs -> action in [-1, 1]).

    Returns (metrics, recording). `recording` holds per-step arrays:
      qpos, qvel      full MuJoCo state, for re-rendering the performance
      key_activation  (T, 88) which keys were sounding
      goal            (T, 88) which keys should have been sounding
      reward          (T,)
    """
    ts = env.reset()
    physics = env.physics  # Fetch after reset: composer can rebuild physics on reset.
    piano = env.task.piano
    frames = {k: [] for k in ("qpos", "qvel", "key_activation", "goal", "reward")}

    def snapshot(reward: float, goal: np.ndarray) -> None:
        frames["qpos"].append(physics.data.qpos.copy())
        frames["qvel"].append(physics.data.qvel.copy())
        frames["key_activation"].append(piano.activation.copy())
        frames["goal"].append(goal)
        frames["reward"].append(reward)

    episode_return = 0.0
    if record:
        snapshot(0.0, np.zeros(piano.n_keys, bool))
    while not ts.last():
        ts = env.step(policy(ts.observation))
        episode_return += float(ts.reward)
        if record:
            snapshot(float(ts.reward), env.task._goal_current[:-1].astype(bool))

    metrics = dict(env.get_musical_metrics(), episode_return=episode_return)
    recording = None
    if record:
        recording = {k: np.asarray(v) for k, v in frames.items()}
        recording["control_timestep"] = np.float32(env.task.control_timestep)
        metrics.update(note_metrics(recording["goal"], recording["key_activation"]))
    return metrics, recording


def save_recording(path: str | Path, recording: dict, metrics: dict, step: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **recording,
        train_step=np.int64(step),
        **{f"metric_{k}": np.float32(v) for k, v in metrics.items()},
    )
    return path
