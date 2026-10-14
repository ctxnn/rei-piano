# 05 · Running on SF Compute Autoresearch (H100s)

How this project uses [Autoresearch](https://autoresearch.sfcompute.com/docs) ("give me a
node"): GPU nodes rented by the minute, driven from a laptop with the `autor` CLI.

## One-time setup

```bash
curl -fsSL https://autoresearch.sfcompute.com/cli.sh | bash   # installs ~/.local/bin/autor
autor login                                                    # browser sign-in, once
autor billing                                                  # balance and runway
```

* Mission page (where a human watches the work, the cost and the updates):
  <https://autoresearch.sfcompute.com/missions/runaway-piano>
* Storage bucket for artifacts (checkpoints, recordings, videos):
  `runaway-piano`. Nodes see it as the connection `storage/runaway-piano`.

## Prices we saw (2026-10-12)

| What | Price | Notes |
|---|---|---|
| `h100-1` node (1 GPU, 14 vCPU, ~250 GiB RAM) | $5.04/h list | $4.03/h weekends (−20%), Fridays −33% |
| `h100-8` node (8 GPU, 112 vCPU, ~1.9 TiB RAM) | $40.32/h list | same discounts |
| Batch job, per H100 | $3.78/h | 25% below nodes, preemptible with checkpoint slots |
| `cpu-8` node | $0.72/h list | no GPU |
| Object storage | first 100 GB free, then $0.10/GB-month | |
| Idle window | a node with nothing running stops itself after 15 min (8x: 20 min) and bills that window | |

## The workflow we use

```bash
autor node create --name piano-bench --chip h100-1 --image jax-0.11-cuda12.9 --mission runaway-piano
bash scripts/push_code.sh piano-bench                 # code -> ~/runaway on the node
autor run piano-bench -d -- bash -c 'cd ~/runaway && bash scripts/node_setup.sh'
autor run piano-bench -d -- bash -c 'cd ~/runaway && .venv/bin/python -m runaway.train --run-name r1 ...'
autor logs piano-bench -f                             # follow output
autor run piano-bench -- python3 ~/runaway/scripts/summarize_runs.py ~/runaway/runs
autor node stop piano-bench                           # billing stops, disk is kept
```

* There is **no SSH**. Every command goes through `autor run`. Commands longer than
  60 s (from an agent) must be **detached** (`-d`), then followed with `autor logs`.
* Only the home directory (`/home/dev`) survives a stop. `apt` packages vanish, which is
  why `scripts/node_setup.sh` re-installs them; the Python venv lives in `~/runaway/.venv`
  and survives.
* A detached training run keeps the node "busy", so it will not idle-stop under it.

## Gotchas we hit (and fixed)

1. **CPU budget vs. what Python sees.** A 1-GPU node gets 14 vCPUs, but `os.cpu_count()`
   reports the whole machine (112). Size worker pools from `$GMN_CPU_LIMIT`.
2. **Thread oversubscription.** The platform presets `OMP_NUM_THREADS=14` (the whole
   budget) for every process. With 13 simulator worker processes that is 13 × 14 threads,
   and simulation ran **2× slower**. `runaway/vec_env.py` now forces 1 thread per worker
   before spawning them. Measured: 882 → 1714 env steps/s with 13 workers.
3. **The node CPU is slower per core than an Apple M5.** One RoboPianist env: 823 steps/s
   on the M5 vs ~260–290 on the node's Xeon Platinum 8480+. Parallel workers make up for it.
4. **`pkill -f <pattern>` inside `autor run ... bash -c '...'`** matches the bash command
   itself and kills it (exit 143). Don't do that.
5. **Tarballs from macOS** carry extended attributes that GNU tar warns about. Use
   `COPYFILE_DISABLE=1 tar --no-xattrs` (done in `scripts/push_code.sh`).
6. **GPU rendering is off by default.** MuJoCo's EGL renderer needs "GPU graphics", which
   SF Compute turns on per organization by ticket. We avoid needing it: training records
   simulator states (`.npz`), and videos are rendered elsewhere (`runaway/render.py`).
7. **The 8×H100 node has 224 CPU threads** (112 cores × 2 hyperthreads, 4 NUMA nodes),
   twice what the docs' table suggests, so each run can have 20 simulators.
8. **4,096-thread limit per container.** Hit twice: (a) unpinned JAX processes create
   ~940 threads each (pools sized to all 224 CPUs) → pin each run with `taskset` to 28
   CPUs (`scripts/launch_runs.sh`); (b) MuJoCo's compiler starts a thread per mesh, and
   160 workers compiling the G1 at once crashed with "Caught an unknown exception" → we set
   `compiler usethread="false"` in `make_env`. Check usage with
   `cat /sys/fs/cgroup/pids.current` (limit: `pids.max`).
9. **Big uploads with `autor cp` can fail** ("request failed" on an 18 MB multi-part
   upload). Have the node download from the source instead (`scripts/get_g1.sh` clones the
   G1 model from GitHub on the node).
10. **Synchronous `autor run` calls time out** around 60–120 s ("request failed"); run
    anything long with `-d` and poll.

## Where things are stored

| Artifact | Where | Size |
|---|---|---|
| Code | git repo (laptop) → copied to each node with `push_code.sh` | ~25 KB |
| Training runs (`runs/<name>/`: metrics, eval recordings, actor checkpoints) | node disk `~/runaway/runs`, exported to bucket `runaway-piano` | ~100–500 MB per run |
| Showcase videos | rendered from recordings, the final ones copied to `outputs/` on the laptop | few MB each |
| Song audio | laptop only (`data/audio/`, git-ignored, copyrighted) | 8 MB |
