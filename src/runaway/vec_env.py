"""Run many copies of the environment in parallel, one per CPU core.

MuJoCo simulates this scene on the CPU at roughly 300-750 steps per second per core.
The learner (on the GPU) can consume data much faster than that, so we keep the GPU
fed by running N environments in N worker processes and stepping them in lockstep:

    main process                     worker k (one per core)
    ------------                     -----------------------
    actions (N, act_dim)  ---pipe--> env_k.step(actions[k])
    obs, reward, ...      <--pipe--- (auto-resets when the song ends)

Every copy plays the same song, so all copies finish their episodes on the same step.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np

from runaway.env import EnvConfig, make_env


_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@dataclass
class StepResult:
    obs: np.ndarray  # (N, obs_dim) observation to act on next (after auto-reset)
    reward: np.ndarray  # (N,)
    terminated: np.ndarray  # (N,) bool: episode ended and the value should NOT bootstrap
    truncated: np.ndarray  # (N,) bool: episode ended because the song ended
    final_obs: np.ndarray  # (N, obs_dim) the true last observation, for the replay buffer
    episode_metrics: list[dict | None]  # per env: F1 etc. when an episode just ended


def _worker(remote, cfg: EnvConfig, seed: int) -> None:
    env = make_env(cfg, seed=seed)
    episode_return = 0.0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                episode_return = 0.0
                remote.send(env.reset().observation)
            elif cmd == "step":
                ts = env.step(data)
                episode_return += float(ts.reward)
                metrics = None
                obs = final_obs = ts.observation
                if ts.last():
                    metrics = dict(env.get_musical_metrics(), episode_return=episode_return)
                    episode_return = 0.0
                    obs = env.reset().observation
                # dm_env convention: discount == 0 on a real termination (e.g. wrong
                # key with wrong_press_termination). The song simply ending keeps
                # discount 1, which we treat as a time-limit truncation.
                terminated = ts.last() and ts.discount == 0.0
                truncated = ts.last() and not terminated
                remote.send((obs, ts.reward, terminated, truncated, final_obs, metrics))
            elif cmd == "specs":
                remote.send((env.observation_spec().shape, env.action_spec().shape))
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class VecEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int, seed: int = 0) -> None:
        ctx = mp.get_context("spawn")  # "fork" + MuJoCo/JAX threads is not safe.
        # Each worker must be single-threaded: we parallelise across processes. Thread
        # pools inherit their size from these variables when NumPy is first imported,
        # so they must be in the environment *before* the workers start. (Some cloud
        # images preset them to the whole CPU budget; with 12 workers x 14 threads the
        # machine thrashed and ran 2x slower.)
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"
        self.num_envs = num_envs
        self._remotes, self._procs = [], []
        for k in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, seed + k), daemon=True)
            p.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(p)
        self._remotes[0].send(("specs", None))
        obs_shape, act_shape = self._remotes[0].recv()
        self.obs_dim, self.act_dim = obs_shape[0], act_shape[0]

    def reset(self) -> np.ndarray:
        for r in self._remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self._remotes])

    def step(self, actions: np.ndarray) -> StepResult:
        for r, a in zip(self._remotes, actions):
            r.send(("step", a))
        obs, rew, term, trunc, final_obs, metrics = zip(*[r.recv() for r in self._remotes])
        return StepResult(
            obs=np.stack(obs),
            reward=np.asarray(rew, dtype=np.float32),
            terminated=np.asarray(term, dtype=bool),
            truncated=np.asarray(trunc, dtype=bool),
            final_obs=np.stack(final_obs),
            episode_metrics=list(metrics),
        )

    def close(self) -> None:
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self._procs:
            p.join(timeout=5)
