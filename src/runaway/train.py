"""Train a policy to play Runaway with SAC and many parallel environments.

    python -m runaway.train --run-name my-run --num-envs 12 --total-env-steps 3000000

Each iteration of the loop:
  1. a CPU (NumPy) copy of the actor picks actions for all N environments
  2. the gradient update is *dispatched* to the GPU (JAX runs it asynchronously)...
  3. ...while the CPU workers simulate one control step of all N environments
  4. the N new transitions are staged, and copied to the GPU replay buffer in batches
So simulation (CPU) and learning (GPU) overlap. The CPU actor copy is refreshed from
the GPU every `actor_sync_every` iterations; being a few updates stale does not matter
for off-policy learning. (Measured on an H100 node, these two tricks removed ~8 ms of
host time per iteration: see docs/04_training.md.)

Outputs, under <out_root>/<run_name>/:
  config.json            every setting of this run
  metrics.jsonl          one JSON line per log/eval event (plot with scripts/plot_metrics.py)
  evals/step_XXXXXXXXX.npz   a recorded deterministic performance (render.py makes a video)
  checkpoints/actor_step_XXXXXXXXX.msgpack   actor weights at that point
  checkpoints/agent_latest.msgpack + progress.json   full agent state, for resuming

Resuming (e.g. after the node stopped because credit ran out): run the same command
again. If checkpoints/agent_latest.msgpack exists, training continues from it. The
replay buffer is not saved (9 GB); it refills with data from the current policy, and
gradient updates pause until it holds `warmstart_steps` transitions again.
"""

from __future__ import annotations

import os

# One thread per process for NumPy's math libraries. Must be set before NumPy is
# imported. The simulator workers already use all the other cores; extra BLAS threads
# here would only fight them. (Some cloud images preset these to the whole CPU budget.)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from flax import serialization  # noqa: E402

from runaway.env import EnvConfig, make_env  # noqa: E402
from runaway.evaluate import play_song, save_recording  # noqa: E402
from runaway.sac import (  # noqa: E402
    SAC, NumpyActor, ReplayBuffer, SACConfig, add_packed_to_buffer, pack_transitions,
)  # fmt: skip
from runaway.vec_env import VecEnv  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    run_name: str = "debug"
    out_root: str = "runs"
    seed: int = 0
    # Parallel simulators, one per CPU core. Leave ~2 cores for the main process and
    # JAX's runtime threads (on a 14-vCPU H100 node: 12).
    num_envs: int = 12
    total_env_steps: int = 3_000_000
    # Uniformly random actions for the first steps, to fill the replay buffer.
    warmstart_steps: int = 5_000
    # Gradient steps per environment step ("update-to-data ratio"). RoboPianist: 1.
    updates_per_env_step: float = 1.0
    replay_capacity: int = 1_000_000
    eval_every: int = 50_000
    log_every: int = 10_000
    # Save actor weights at every eval, for "how it improved" videos.
    save_actor_every_eval: bool = True
    # Copy new transitions to the GPU replay buffer every this many loop iterations,
    # as one packed array (each host->GPU copy has ~0.6 ms of fixed latency).
    store_every: int = 4
    # Refresh the CPU copy of the actor from the GPU every this many loop iterations.
    actor_sync_every: int = 4
    # Save the full agent every this many env steps (atomic write), so a stopped node
    # costs at most this much progress.
    checkpoint_every: int = 250_000
    # Continue from checkpoints/agent_latest.msgpack if it exists.
    resume: bool = True
    # Warm start: initialise the agent (networks, optimisers, temperature) from another
    # run's checkpoint, e.g. runs/r2_g1_g088_s0/checkpoints/agent_latest.msgpack. The new
    # run counts its own steps from 0 and skips the random-action warm-up; observation and
    # action sizes must match (same embodiment).
    init_from: str | None = None


def _jsonable(x):
    if dataclasses.is_dataclass(x):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    return x


class Logger:
    def __init__(self, path: Path) -> None:
        self._f = open(path, "a", buffering=1)

    def log(self, **kv) -> None:
        kv = {k: (float(v) if isinstance(v, (np.floating, jnp.ndarray)) else v) for k, v in kv.items()}
        self._f.write(json.dumps(kv) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()),
              flush=True)  # fmt: skip


def train(cfg: TrainConfig) -> None:
    out = Path(cfg.out_root) / cfg.run_name
    (out / "evals").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(_jsonable(cfg), indent=2))
    logger = Logger(out / "metrics.jsonl")
    print(f"JAX devices: {jax.devices()}  |  run dir: {out}")

    venv = VecEnv(cfg.env, cfg.num_envs, seed=cfg.seed * 1000)
    eval_env = make_env(cfg.env, seed=cfg.seed * 1000 + 999)
    agent = SAC.create(cfg.sac, venv.obs_dim, venv.act_dim, seed=cfg.seed)
    buffer = ReplayBuffer.create(cfg.replay_capacity, venv.obs_dim, venv.act_dim)
    updates_per_iter = max(1, round(cfg.updates_per_env_step * cfg.num_envs))
    rng = np.random.default_rng(cfg.seed)

    ckpt_path = out / "checkpoints" / "agent_latest.msgpack"
    progress_path = out / "checkpoints" / "progress.json"
    progress = {"step": 0, "next_eval": 0, "next_log": cfg.log_every,
                "next_checkpoint": cfg.checkpoint_every, "hours": 0.0}  # fmt: skip
    if cfg.resume and ckpt_path.exists() and progress_path.exists():
        agent = serialization.from_bytes(agent, ckpt_path.read_bytes())
        progress = json.loads(progress_path.read_text())
        logger.log(event="resume", step=progress["step"])
    elif cfg.init_from:
        agent = serialization.from_bytes(agent, Path(cfg.init_from).read_bytes())
        logger.log(event="warm_start", init_from=cfg.init_from)

    def save_checkpoint(step: int, hours: float) -> None:
        """Write the agent and loop counters; tmp file + rename = never half-written."""
        for path, data in ((ckpt_path, serialization.to_bytes(agent)),
                           (progress_path, json.dumps(dict(progress, step=step, hours=hours)).encode())):  # fmt: skip
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)

    cpu_actor = NumpyActor(agent.actor.params)

    def evaluate(step: int) -> None:
        t0 = time.time()
        policy = NumpyActor(agent.actor.params)  # latest weights, deterministic actions
        metrics, rec = play_song(eval_env, lambda o: policy(o[None])[0])
        save_recording(out / "evals" / f"step_{step:09d}.npz", rec, metrics, step)
        if cfg.save_actor_every_eval:
            (out / "checkpoints" / f"actor_step_{step:09d}.msgpack").write_bytes(
                serialization.to_bytes(agent.actor.params)
            )
        logger.log(event="eval", step=step, eval_seconds=time.time() - t0,
                   **{f"eval/{k}": v for k, v in metrics.items()})  # fmt: skip

    obs = venv.reset()
    it = 0
    step, next_eval, next_log = progress["step"], progress["next_eval"], progress["next_log"]
    next_checkpoint = progress["next_checkpoint"]
    hours_before = progress["hours"]  # time spent in earlier (interrupted) sessions
    stored = 0  # transitions in the replay buffer (it starts empty, also after a resume)
    train_infos, episode_stats = None, []
    staged = []  # transitions waiting to be copied to the GPU buffer
    t_start = t_last = time.time()
    steps_at_last = step
    # Wall-clock milliseconds spent in each phase of the loop (host side), to find
    # bottlenecks. GPU work is asynchronous, so its cost shows up where the host
    # first *waits* for a result (usually "act").
    phase_ms = {"act": 0.0, "learn": 0.0, "simulate": 0.0, "store": 0.0}
    phase_iters = 0

    while step < cfg.total_env_steps:
        if step >= next_eval:
            evaluate(step)
            next_eval += cfg.eval_every
        if step >= next_checkpoint:
            progress.update(next_eval=next_eval, next_log=next_log, next_checkpoint=next_checkpoint + cfg.checkpoint_every)
            save_checkpoint(step, hours_before + (time.time() - t_start) / 3600)
            next_checkpoint += cfg.checkpoint_every

        t0 = time.perf_counter()
        # 1) Act.
        if step < cfg.warmstart_steps and not cfg.init_from:
            actions = rng.uniform(-1, 1, (cfg.num_envs, venv.act_dim)).astype(np.float32)
        else:
            if it % cfg.actor_sync_every == 0:
                cpu_actor.load(agent.actor.params)  # waits for the last GPU update
            actions = cpu_actor(obs, rng)
        t1 = time.perf_counter()

        # 2) Learn (asynchronous: returns immediately, the GPU keeps working).
        if stored >= cfg.warmstart_steps:
            agent, train_infos = agent.update(buffer, updates_per_iter)
        t2 = time.perf_counter()

        # 3) Simulate.
        res = venv.step(actions)
        t3 = time.perf_counter()

        # 4) Store. `final_obs` is the real next observation even when an env reset.
        staged.append(pack_transitions(obs, actions, res.reward,
                                       1.0 - res.terminated, res.final_obs))  # fmt: skip
        if len(staged) >= cfg.store_every:
            buffer = add_packed_to_buffer(buffer, jnp.asarray(np.concatenate(staged)))
            stored += sum(len(x) for x in staged)
            staged = []
        obs = res.obs
        t4 = time.perf_counter()
        for name, dt in zip(phase_ms, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            phase_ms[name] += 1000 * dt
        phase_iters += 1
        step += cfg.num_envs
        it += 1
        episode_stats += [m for m in res.episode_metrics if m is not None]

        if step >= next_log:
            now = time.time()
            kv = dict(event="train", step=step,
                      steps_per_sec=(step - steps_at_last) / (now - t_last),
                      hours=hours_before + (now - t_start) / 3600)  # fmt: skip
            kv.update({f"time_ms/{k}": v / phase_iters for k, v in phase_ms.items()})
            phase_ms = dict.fromkeys(phase_ms, 0.0)
            phase_iters = 0
            if train_infos is not None:
                kv.update({f"train/{k}": float(v) for k, v in train_infos.items()})
            if episode_stats:
                for k in episode_stats[0]:
                    kv[f"rollout/{k}"] = float(np.mean([m[k] for m in episode_stats]))
                episode_stats = []
            logger.log(**kv)
            t_last, steps_at_last = now, step
            next_log += cfg.log_every

    evaluate(step)
    progress.update(next_eval=next_eval, next_log=next_log, next_checkpoint=next_checkpoint)
    save_checkpoint(step, hours_before + (time.time() - t_start) / 3600)
    venv.close()
    print(f"done: {step} env steps in {(time.time() - t_start) / 3600:.2f} h")


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
