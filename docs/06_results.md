# 06 · Results

Two scores, both from one deterministic performance of the whole clip:

* **Per-step F1** (RoboPianist's metric): overlap of keys down vs keys wanted, every 50 ms,
  averaged (1.0 = perfect; pressing nothing = 0.175).
* **Note F1** (added 2026-10-12 evening): each strike must start on the right key within
  ±100 ms; every extra key press counts against precision. **This is how it sounds.**

> **Correction.** Until the evening of 2026-10-12 these docs reported only per-step F1 and
> called the G1 humanoid (0.876) "playing well". It does not: it makes 225–366 key presses
> for 44 strikes (finger jitter and brushed neighbouring keys), note F1 ≤ 0.27. The user
> heard it before the metric caught it. Per-step F1 forgives 50 ms blips; ears do not.
>
> **Update (round 7).** The main cause was a bug: the G1 policy could not observe its own
> finger joints (see Round 6). With that fixed, the G1 reaches note F1 0.905 (Round 7).

### Note-level re-scoring of every run (best evaluation per run)

| run | best note F1 | at step | presses / strikes | per-step F1 there | best per-step F1 |
|---|---|---|---|---|---|
| r1_g084_s0 | **0.989** | 1,000,000 | 45 / 44 | 1.000 | 1.000 |
| r1_g084_s1 | **0.936** | 1,000,000 | 50 / 44 | 0.986 | 0.986 |
| r1_g088_45s_s0 | **1.000** | 1,000,000 | 33 / 33 | 1.000 | 1.000 |
| r1_g088_b1024_s0 | **0.989** | 900,000 | 45 / 44 | 0.998 | 0.998 |
| r1_g088_s0 | **0.957** | 1,000,000 | 48 / 44 | 0.991 | 0.995 |
| r1_g088_s1 | **0.967** | 1,100,000 | 47 / 44 | 0.996 | 0.996 |
| r1_g092_s0 | **0.978** | 900,000 | 46 / 44 | 0.999 | 0.999 |
| r1_g092_s1 | **0.978** | 1,200,000 | 46 / 44 | 0.997 | 0.998 |
| r2_g1_g084_s0 | **0.268** | 5,500,000 | 225 / 44 | 0.862 | 0.862 |
| r2_g1_g088_b1024_s0 | **0.209** | 3,900,000 | 339 / 44 | 0.714 | 0.733 |
| r2_g1_g088_s0 | **0.269** | 5,500,000 | 253 / 44 | 0.876 | 0.876 |
| r2_g1_g088_s1 | **0.228** | 5,900,000 | 307 / 44 | 0.774 | 0.822 |
| r2_g1_g088_s2 | **0.218** | 5,300,000 | 305 / 44 | 0.712 | 0.784 |
| r2_g1_g092_s0 | **0.205** | 5,700,000 | 346 / 44 | 0.768 | 0.798 |
| r2_g1_g092_s1 | **0.219** | 2,200,000 | 267 / 44 | 0.702 | 0.759 |
| r2_g1_g092_s2 | **0.210** | 5,500,000 | 366 / 44 | 0.779 | 0.790 |

The tables below give the per-step numbers as originally logged. Evaluations every
100k env steps. Raw data: `s3://runaway-piano/round1/runs.tar` (org storage bucket).

## Round 1: Shadow Hands on sliders (RoboPianist setup), 60 s clip

8 runs on one 8×H100 node (`piano-r1`), 20 simulators per run, ~1,500 env steps/s per run.
Stopped at ~1.25M steps (≈ 14 min) because every run had converged. Cost ≈ $25.

| run | discount | seed | variant | best eval F1 | at step | first ≥ 0.95 | first ≥ 0.99 | final eval F1 | steps trained |
|---|---|---|---|---|---|---|---|---|---|
| r1_g084_s0 | 0.84 | 0 | – | **1.000** | 1,000,000 | 500,000 | 1,000,000 | 0.993 | 1,240,000 |
| r1_g084_s1 | 0.84 | 1 | – | 0.986 | 1,000,000 | 600,000 | never | 0.950 | 1,250,000 |
| r1_g088_45s_s0 | 0.88 | 0 | 45 s clip | **1.000** | 1,000,000 | 500,000 | 600,000 | 0.981 | 1,270,000 |
| r1_g088_b1024_s0 | 0.88 | 0 | batch 1024, UTD 0.25 | 0.998 | 900,000 | 500,000 | 700,000 | 0.997 | 1,240,000 |
| r1_g088_s0 | 0.88 | 0 | – | 0.995 | 1,100,000 | 500,000 | 1,000,000 | 0.992 | 1,240,000 |
| r1_g088_s1 | 0.88 | 1 | – | 0.996 | 1,100,000 | 700,000 | 1,100,000 | 0.988 | 1,240,000 |
| r1_g092_s0 | 0.92 | 0 | – | 0.999 | 900,000 | 600,000 | 900,000 | 0.987 | 1,250,000 |
| r1_g092_s1 | 0.92 | 1 | – | 0.998 | 1,000,000 | 300,000 | 500,000 | 0.997 | 1,240,000 |

**What we learned**

* **The song is easy for this setup.** Every configuration reached F1 ≥ 0.95 within
  300k–700k steps (4–8 minutes) and ≥ 0.99 by ~1M. Runaway's piano is one note at a time,
  slow, in a 20 cm range. Compare RoboPianist's average of 0.79 on harder pieces after 5M.
* **Discount 0.84–0.92 all work**; no clear winner at this sample size. The paper's warning
  is about 0.96+.
* **The 4× bigger batch with 4× fewer updates** matched the default (0.998), with fewer GPU
  kernel launches: a lead for making training cheaper.
* **Eval F1 wobbles by ~0.01–0.04 between checkpoints** after convergence (e.g. 1.000 →
  0.993). Keep the *best* checkpoint for showcases, not the last.
* What remains imperfect in non-perfect runs: occasional early/late presses by one control
  step (50 ms) at note boundaries, visible in the dashboard's piano roll.

## Round 2: Unitree G1 humanoid (RL controls shoulders + elbows)

8 runs, same node. Trained to 3M steps (≈ 40 min), then resumed from the saved
checkpoints to **6M steps** (another ≈ 40 min; ~1,350 env steps/s per run). The arm joints
replace the sliders, so the policy has to learn to swing the arms to carry the hands.
First attempt (arm range ±1.2 rad) learned nothing in 300k steps; with ±0.4 rad and the
piano at 0.75 m every run learns. Raw data:
`s3://runaway-piano/round2/runs.tar` (includes the failed first attempt).

| run | discount | seed | best F1 by 3M | best F1 by 6M | at step | precision / recall at best | first ≥ 0.80 | F1 @ 6M |
|---|---|---|---|---|---|---|---|---|
| r2_g1_g088_s0 | 0.88 | 0 | 0.800 | **0.876** | 5,500,000 | 0.91 / 0.92 | 4,500,000 (0.7998 at 2.8M) | 0.831 |
| r2_g1_g084_s0 | 0.84 | 0 | 0.765 | **0.862** | 5,500,000 | 0.91 / 0.90 | 4,300,000 | 0.815 |
| r2_g1_g088_s1 | 0.88 | 1 | 0.735 | **0.822** | 5,800,000 | 0.88 / 0.88 | 5,800,000 | 0.705 |
| r2_g1_g092_s0 | 0.92 | 0 | 0.721 | **0.798** | 5,400,000 | 0.84 / 0.87 | never | 0.773 |
| r2_g1_g092_s2 | 0.92 | 2 | 0.741 | **0.790** | 5,900,000 | 0.84 / 0.85 | never | 0.737 |
| r2_g1_g088_s2 | 0.88 | 2 | 0.694 | **0.784** | 5,700,000 | 0.87 / 0.84 | never | 0.725 |
| r2_g1_g092_s1 | 0.92 | 1 | 0.704 | **0.759** | 5,200,000 | 0.83 / 0.83 | never | 0.674 |
| r2_g1_g088_b1024_s0 | 0.88 (batch 1024) | 0 | 0.727 | **0.733** | 5,600,000 | 0.82 / 0.82 | never | 0.694 |

Video: `outputs/showcase/g1_5.5M_jittery_noteF1_0.27*.mp4` (bucket: `showcase/`), camera
`humanoid_front_left`. It reaches the right region of the keyboard and hits 40 of 44
strikes on time, but it **does not sound right**: 253 presses for 44 strikes.

**Why the G1 sounds wrong (error analysis of r2_g1_g088_s0 @ 5.5M):**

| extra presses | count |
|---|---|
| keys the song never uses (D6, F6, C6, B5, G5…: neighbours of the real notes) | 129 |
| re-presses of a key that should stay down (chatter) | 70 |
| a song key at the wrong time | 54 |

74% of all presses last ≤ 100 ms. The hand **jitters**: small shoulder/elbow wobbles are
amplified along the ~40 cm arm into fingertip movements of millimetres, enough to tap a
neighbouring key or bounce a held one. Neither the reward nor per-step F1 penalised this
much. Candidate fixes (to do): action smoothing / an action-rate penalty on the arm, a
reward term per extra key press (not just "any wrong key now"), stiffer/damped arm servos,
and joint- vs Cartesian (IK) arm commands.

**What we learned**

* **Arm control makes the task much harder**: per-step F1 0.876 after 6M steps but note F1
  only 0.27 (it doesn't sound right), vs note F1 0.99–1.00 after 1M with RoboPianist's sliders.
* **Exploration range decides whether learning starts at all** (±1.2 rad: nothing in 300k
  steps; ±0.4 rad: learns), measured with random actions before training.
* **The policy fixed the idle-arm problem itself**: at 1.2M the left hand rested on the
  keys (724 steps of wrong notes); by 2.8M it lifts the left arm clear of the keyboard.
* **Seed variance is large**: 0.73–0.88 across 8 runs with near-identical settings.
  Any comparison (algorithms, settings) needs ≥ 3 seeds per arm.
* **Discount**: 0.84–0.88 did best here (top 2 runs); 0.92 never reached 0.80.
* **Big batch (1024, UTD 0.25) helped in round 1 but plateaued here** (0.73): fewer
  gradient steps per sample seem to matter once the task is hard.
* **Eval F1 at a single checkpoint is noisy** (e.g. 0.876 best vs 0.831 at 6M for the same
  run): always keep the best checkpoint, and evaluate several episodes if comparing.

## Round 3: fixing the G1's jitter (did not fix it)

8 runs warm-started from the two best round-2 checkpoints (`r2_g1_g088_s0`,
`r2_g1_g084_s0`, both at 6M), 3M more steps each, with the anti-jitter terms of
[07_math.md §3.4](07_math.md): A = extra-press penalty 0.5, B = 2.0, C = A + arm filter
0.5 + rate penalty 1, D = B + arm filter 0.5 + rate penalty 5. Sweep file:
`sweeps/round3_g1_fix.txt`.

First observation, before any new training (the warm-start checkpoint evaluated in the
new environment): the arm filter alone cuts presses from 272 to 188 for 44 strikes
(note F1 0.26 → 0.34) on `r2_g1_g088_s0`.

| run | variant | best note F1 | at 3M: note F1, presses / 44 | per-step F1 at 3M |
|---|---|---|---|---|
| r3_g084_A | penalty 0.5 | **0.380** (at 3M) | 0.380, 172 | 0.903 |
| r3_g088_C | 0.5 + filter + rate 1 | 0.371 (at 0) | 0.187, 191 | 0.669 |
| r3_g088_A | penalty 0.5 | 0.348 (at 3M) | 0.348, 180 | 0.844 |
| r3_g084_C | 0.5 + filter + rate 1 | 0.347 | 0.237, 226 | 0.667 |
| r3_g088_D | 2.0 + filter + rate 5 | 0.336 (at 0) | 0.154, 73 | 0.363 |
| r3_g088_B | penalty 2.0 | 0.259 (at 0) | 0.086, 119 | 0.373 |
| r3_g084_D | 2.0 + filter + rate 5 | 0.240 (at 0) | 0.047, 171 | 0.311 |
| r3_g084_B | penalty 2.0 | 0.214 (at 0) | 0.000, 98 | 0.193 |

**What we learned**
* A **strong** extra-press penalty (2.0) makes the policy stop playing (per-step F1 falls
  to 0.2–0.37): pressing anything became too risky. A **mild** one (0.5) improves slowly:
  272 → 172 presses, note F1 0.26 → 0.38 in 3M steps, still rising.
* The arm filter helps at once but the rate penalty on top did not add anything in 3M steps.
* Conclusion: reward tweaks on a policy that learned to jitter are not enough. Next
  candidates (TODO.md §1): hand-position commands through IK (slider-like control with real
  arms), or training from scratch with the filter + mild penalty on from the start.

Round 3 data is on `piano-r1`'s disk (`~/runaway/runs/r3_*`), not yet in the bucket.

## Round 4: G1 with hand-position (IK) control, from scratch (worse)

The policy sets wrist positions; inverse kinematics drives shoulders and elbows
([07_math.md §6.1](07_math.md)). 8 runs, 4M steps: pure IK ×2 seeds, + wrist-target filter
×2, + filter + extra-press penalty 0.5 ×3, + penalty only ×1. Sweep file:
`sweeps/round4_g1_ik.txt`.

| variant | best note F1 | presses / 44 at 4M |
|---|---|---|
| penalty 0.5 (r4_ik_p_s0) | **0.360** | 133 |
| filter + penalty (3 seeds) | 0.088–0.201 | 140–203 |
| pure IK (2 seeds) | 0.130–0.155 | 431–462 |
| filter (2 seeds) | 0.108–0.148 | 485–489 |

IK alone made it *worse* (430–490 presses). That pointed away from the controller.

### Root cause (found after round 4, by measuring the simulator, not by training)

With a **constant** command, the G1's fingertip crept sideways ~6 mm in 4 s, while the
slider hands stayed still (0.2 mm). The reason: with the piano at 0.75 m, the G1's neutral
pose put the fingertips **on the keys** (1–10 mm above the key tops, 8 fingers in contact);
RoboPianist's hands hover 37–46 mm above. Resting fingers slide and brush keys, and every
small motion becomes a press. The 0.75 m height was chosen in round 2 to make random
exploration hit keys more often (a metric that looked good and caused this). Stiffer arm
servos did not help (tested 1×–25×). With the piano at 0.70 m and IK (whose wrist z range
reaches the keys) the G1 matches the slider hands at rest:

| at the neutral command | sliders | G1, rounds 2–4 | G1, round 5 setup |
|---|---|---|---|
| fingertips above keys | 37–46 mm | 1–10 mm | 40–50 mm |
| sideways creep in 4 s | 0.2 mm | 5.8 mm | 0.2 mm |
| random exploration near keys | 55% | 52% | 49% |

## Round 5: G1 + IK + hands hovering (stopped at ~1M: learned to "dab")

8 runs from scratch: pure ×3 seeds, + filter ×2, + penalty ×2, + both ×1
(`sweeps/round5_g1_ik_hover.txt`). At 700–800k steps: per-step F1 0.18–0.40, note F1 ≤ 0.07,
527–571 presses for 44 strikes (slider hands at the same point: ~0.95, 58 presses).

**Second root cause.** The G1's presses were mostly 1-step blips on the right keys and
their neighbours. Test: hold a constant "lower the hand" command for 3 s.
RoboPianist's slider **cannot** lower the hand onto the keys (0 key presses), so its
policies must press with individual fingers. The G1's wrist (z range ±12 cm) pushed the
whole hand into the keyboard: 4–6 keys at once. The policy learned to play by dabbing the
whole hand. Fix: wrist range x ±3 cm, z ±3 cm, so the lowest wrist leaves the fingertips just
above the keys (0 presses at full-down; flexing a finger presses a key). Round 5 stopped
early to start round 6 on the same node.

## Round 6: G1 + IK + hovering + finger-only pressing (stopped at ~650k)

Same 8 variants as round 5 (`sweeps/round6_g1_ik_fingers.txt`). At 600k: 62–431 presses,
note F1 ≤ 0.09; 60% one-step blips, all made by fingertips (contact replay). The slider
hands never exceeded ~120 presses at any point of training.

### The real bug: the G1 policy could not see its fingers

The G1 environment's observation was 1,076 numbers: goal (979) + piano (89) + **only the 8
arm joints**. The two hands' finger joint angles (2 × 21) were missing; the slider setup has
them (2 × 23). Cause: `humanoid.py` attached each Shadow Hand to the G1 with a raw MJCF
`site.attach(hand.mjcf_model)` instead of composer's `G1.attach(hand, attach_site=...)`.
The physics and the finger actions worked, so nothing crashed, but composer never
registered the hands as child entities and dropped their observables. **Every G1 round
(2–6) trained a policy that was blind to its own fingers.** This alone can explain the
flicking and chatter, and it means the earlier G1 "root causes" (resting height, whole-hand
dabbing) were real but secondary. Fixed, with a regression test
(`tests/test_humanoid_observations.py`).

## Round 7: G1 with finger observations: **the G1 plays** (note F1 up to 0.905)

5 IK runs (3 seeds, + filter, + penalty) and 3 joint-control runs, piano 0.70 m, 4M steps
each, from scratch (`sweeps/round7_g1_obsfix.txt`). ~0.9 h on one 8×H100 node.

| run | best note F1 | at step | presses / strikes | note recall | final (4M) note F1 | final presses | final per-step F1 | first step with note F1 ≥ 0.5 |
|---|---|---|---|---|---|---|---|---|
| r7_ik_p_s0 (IK + extra-press penalty 0.5) | **0.905** | 3,600,000 | 51 / 44 | 0.98 | 0.724 | 72 | 0.961 | 2,100,000 |
| r7_joint_s2 | **0.871** | 3,500,000 | 57 / 44 | 1.00 | 0.822 | 63 | 0.983 | 1,500,000 |
| r7_ik_f_s0 (IK + action filter 0.5) | **0.863** | 3,600,000 | 58 / 44 | 1.00 | 0.804 | 63 | 0.987 | 1,900,000 |
| r7_joint_s0 | **0.830** | 3,900,000 | 62 / 44 | 1.00 | 0.721 | 67 | 0.962 | 1,700,000 |
| r7_ik_s1 | **0.827** | 3,600,000 | 60 / 44 | 0.98 | 0.750 | 68 | 0.982 | 1,700,000 |
| r7_ik_s0 | **0.811** | 3,800,000 | 62 / 44 | 0.98 | 0.629 | 96 | 0.943 | 1,700,000 |
| r7_joint_s1 | **0.771** | 3,800,000 | 65 / 44 | 0.95 | 0.677 | 83 | 0.959 | 1,800,000 |
| r7_ik_s2 | **0.746** | 4,000,000 | 74 / 44 | 1.00 | 0.746 | 74 | 0.983 | 2,100,000 |

What it means:

* **Fixing the observation was the fix.** Same reward, same algorithm, same budget: rounds
  2–6 never passed note F1 0.36; every round-7 run passed 0.74. At 600k the round-7 runs were
  still at note F1 0.11–0.17 (230–340 presses), so the G1 needs ~3× the steps of the slider
  hands (which reached 0.94–1.0 at ~1M), but it gets there.
* **Almost every note is hit** (recall 0.95–1.00). What remains are 7–30 extra presses per
  60 s, so precision is the gap to the slider hands.
* **Joint vs IK: no clear difference** with 3 seeds each (best note F1 0.77–0.87 vs
  0.75–0.83). The IK runs with a filter or the extra-press penalty were the best two IK runs,
  but with one seed each this is a hint, not a result.
* **Evaluations are noisy** between checkpoints (the best run: 0.905 at 3.6M, 0.724 at 4M).
  Keep the best checkpoint by note F1, not the last one. Several runs were still improving
  at 4M; longer training would likely help.
* Video: `outputs/showcase/g1_3.6M_noteF1_0.91*.mp4` (also `s3://runaway-piano/showcase/`).
* Raw data for rounds 3–7: `s3://runaway-piano/round3-7/runs.tar` (3.8 GB).

## Cost summary (2026-10-12)

| Item | Time | Cost |
|---|---|---|
| 1×H100 benchmark node | 41 min | $2.8 |
| 8×H100 node: round 1 + round 2 (6M) + wakes | ~2 h | ~$75 |
| 8×H100 node: rounds 3–7 + wakes | | ~$133 |
| Storage (≈ 6.3 GB in bucket) | | |
| **Total** (per `autor billing`) | | **$218.37** |
