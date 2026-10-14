import time, warnings, numpy as np, sys
warnings.filterwarnings("ignore")
from runaway.env import EnvConfig, make_env
from runaway.vec_env import VecEnv
if __name__ == "__main__":
    env = make_env(EnvConfig(), seed=0); ts = env.reset()
    print("obs", ts.observation.shape, "act", env.action_spec().shape)
    ts = env.step(np.zeros(env.action_spec().shape, np.float32))
    print("reward terms", env.task.reward_fn.compute_reward_terms(env.physics) if hasattr(env.task.reward_fn,'compute_reward_terms') else ts.reward)
    for n in [int(x) for x in sys.argv[1:]] or [1, 4, 8]:
        v = VecEnv(EnvConfig(), n); v.reset(); t = time.time(); steps = 200
        for _ in range(steps): v.step(np.random.uniform(-1, 1, (n, v.act_dim)).astype(np.float32))
        dt = time.time() - t; print(f"N={n}: {n*steps/dt:.0f} steps/s total"); v.close()
