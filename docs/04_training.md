# 04 · Training: algorithm, settings, and how to run it fast

Code: [`src/runaway/sac.py`](../src/runaway/sac.py) (agent),
[`src/runaway/train.py`](../src/runaway/train.py) (loop),
[`scripts/launch_runs.sh`](../scripts/launch_runs.sh) (many runs on one node).

## The algorithm: SAC with DroQ-style critics

* **Actor:** MLP 3×256, GELU → mean and log-std of a Gaussian; action = tanh(sample).
* **Critics:** 2 Q-networks (3×256, GELU) with **LayerNorm + 1% dropout** (the DroQ
  trick: stabilises value learning so many updates per sample are safe).
* **Targets:** soft Bellman target with the *min* of two slow-moving target critics
  (τ = 0.005) minus the entropy term.
* **Temperature α:** learned so the policy's entropy stays near −0.5 × action_dim.
* **Discount γ:** 0.84–0.92 (low on purpose: piano timing is about *now*).
* **Optimiser:** Adam, lr 3e-4 for everything; batch 256; 1 gradient step per env step.

## The training loop, and why it is fast

```
  CPU workers (20 processes)         main process                      GPU
  ──────────────────────────         ────────────                      ───
                                     actions = NumPy actor(obs)
                                     launch 20 SAC updates ─────────▶  scan over 20 updates
  step 20 pianos in parallel ◀────── send actions                     (runs concurrently)
  ──────── obs, rewards ───────────▶ stage transitions
                                     every 4 iterations: one packed
                                     copy into the GPU replay buffer ─▶ 1M-transition buffer
```

Measured on SF Compute (one loop iteration, 20 environments, H100 + 28 CPU threads):
act 3.0 ms · launch updates 1.8 ms · simulate 7.2 ms · store 0.3 ms → **~1,600 env steps/s
per run**. 3M steps ≈ 32 min.

The optimisations, in the order we found them:

| Change | Effect |
|---|---|
| Force 1 math-library thread per process (`OMP_NUM_THREADS=1`) | simulation 2× faster |
| NumPy F1 metric instead of per-step scikit-learn | −42% simulation time |
| Batch transitions into one packed GPU copy every 4 iterations | store 3.4 → 0.5 ms |
| Pin each run to its own CPU block (`taskset`) on the 8-GPU node | fixed a thread-limit crash; ~3× per-run speed vs. the 1-GPU node |
| *Tried and reverted:* unrolling the update scan | host time 4 → 14 ms (more kernel launches) |

## Running it

Locally (CPU, slow but fine for a smoke test):

```bash
.venv/bin/python -m runaway.train --run-name smoke --num-envs 8 --total-env-steps 20000
```

On an SF Compute 8×H100 node, 8 runs at once:

```bash
bash scripts/push_code.sh piano-r1
autor run piano-r1 -d -- bash -c 'cd ~/runaway && bash scripts/node_setup.sh'
autor run piano-r1 -d --env COMMON_ARGS="--total-env-steps 3000000 --eval-every 100000 --num-envs 20" \
  -- bash -c 'cd ~/runaway && bash scripts/launch_runs.sh sweeps/round1.txt'
autor api post /nodes/piano-r1/endpoints --data '{"port": 8000}'   # live dashboard URL
```

## All settings (defaults in code)

| Setting | Default | Notes |
|---|---|---|
| `--num-envs` | 12 | 20 on the 8×H100 node (28 CPU threads per run) |
| `--total-env-steps` | 3,000,000 | ≈ 2,480 performances of the 60 s clip |
| `--warmstart-steps` | 5,000 | random actions to seed the replay buffer |
| `--updates-per-env-step` | 1.0 | RoboPianist's ratio |
| `--replay-capacity` | 1,000,000 | 9 GB on the GPU |
| `--eval-every` | 50,000 | 100,000 in sweeps; each eval records a performance |
| `--sac.discount` | 0.88 | swept 0.84 / 0.88 / 0.92 in round 1 |
| `--sac.batch-size` | 256 | |
| `--store-every`, `--actor-sync-every` | 4, 4 | loop iterations between GPU copies |

## Outputs of a run

`runs/<name>/`: `config.json`, `metrics.jsonl` (train lines every `log_every` steps with
losses, entropy, temperature, Q, rollout F1, phase timings; eval lines with F1/precision/
recall/return), `evals/step_*.npz` (recorded performances), `checkpoints/actor_step_*.msgpack`.
