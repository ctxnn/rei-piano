"""Time each part of the learner loop on this machine's accelerator (no simulation)."""
import time
import jax, jax.numpy as jnp, numpy as np
from runaway.sac import SAC, SACConfig, ReplayBuffer, add_to_buffer

OBS, ACT, N = 1114, 39, 13


def timeit(name, fn, reps=50):
    out = fn(); jax.block_until_ready(out)  # compile
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    print(f"{name:45s} {1000 * (time.perf_counter() - t) / reps:8.3f} ms")
    return out


for cap in (100_000, 1_000_000):
    buf = ReplayBuffer.create(cap, OBS, ACT)
    obs = jnp.zeros((N, OBS)); act = jnp.zeros((N, ACT)); r = jnp.zeros(N)
    state = {"buf": buf}

    def add():
        state["buf"] = add_to_buffer(state["buf"], obs, act, r, r, obs)
        return state["buf"].ptr
    timeit(f"add_to_buffer (capacity {cap:,})", add)
    for _ in range(300):
        add()
    for batch in (256, 1024):
        agent = SAC.create(SACConfig(batch_size=batch), OBS, ACT)
        for k in (1, 6, 13):
            timeit(f"update x{k} batch {batch} (capacity {cap:,})", lambda: agent.update(state["buf"], k)[1]["critic_loss"])
    timeit("sample_actions (13 envs)", lambda: agent.sample_actions(obs)[1])
    timeit("sample_actions + np.asarray (host copy)", lambda: np.asarray(agent.sample_actions(np.zeros((N, OBS), np.float32))[1]))
