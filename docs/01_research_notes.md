# 01 · Research notes: how robots learn to play piano

This is the literature review done before writing any code (2026-10-12). It explains
*why* the project is built the way it is. If you only read one section, read
[Decisions this led to](#decisions-this-led-to).

## The core idea in one paragraph

A piano piece is a timeline of which keys must be down at each moment. In simulation
(MuJoCo) we give a pair of robot hands a reward each time step for having exactly the
right keys pressed, plus "shaping" rewards that pull fingertips toward the keys that
should be pressed next. A reinforcement-learning (RL) algorithm then learns, by trial
and error over millions of simulated steps, a *policy*: a neural network that looks at
the hands, the piano and the upcoming notes, and outputs joint targets for every finger.
One policy is trained per song. Nobody programs the fingering or the hand motion.

## Sources and what each one taught us

### RoboPianist (Zakka et al., CoRL 2023) — the foundation
Paper: <https://arxiv.org/abs/2304.04150> · Code: <https://github.com/google-research/robopianist>

* **Setup.** Two Shadow Dexterous Hands (MuJoCo Menagerie models, 22 actuators each) +
  a full 88-key piano whose keys are hinge joints with springs. A key "sounds" when it is
  within 0.5° of fully pressed. Physics at 500 Hz (actually 5 ms steps), **policy at
  20 Hz** (0.05 s per action).
* **Observation.** Hand joint angles, forearm positions, key states, and the *goal*: a
  piano roll for the current step and the next L steps (lookahead), plus which fingers
  should be used (from human fingering annotations).
* **Action.** Target joint angles for both hands (44) + sustain pedal (1).
* **Reward.** `key_press` (0.5 × right keys down + 0.5 × no wrong key) + `fingering`
  (fingertip close to the key it is annotated to press) − 0.005 × `energy`.
* **Algorithm.** DroQ (= SAC whose critics use dropout + LayerNorm). Adam 3e-4, 3×256
  MLPs, batch 256, 1 gradient update per env step, 5M steps per song, ~5 h on old
  K80-era hardware.
* **Results.** Average F1 0.79 on their 12-song benchmark (MPC baseline: 0.43).
* **Ablations that matter to us:**
  * Without fingering information, F1 stays at **zero**: the fingering shaping reward is
    essential.
  * Discount **0.84–0.92** works best; 0.99 makes agents conservative (they skip notes).
  * **20 Hz** control is the sweet spot (100 Hz: horizon too long; 10 Hz: too coarse).
  * A constant penalty for wrong keys is needed, otherwise hands hover over the keys.
  * Reduced action space (disable a few DoFs, limit the thumb) learns faster.
  * PPO needed ~10× more samples than the off-policy methods.
  * One multi-song policy does not work well: F1 drops from 0.7 (1 song) to ~0 (16 songs).

### RP1M (Zhao et al., CoRL 2024) — no fingering annotations needed
Paper: <https://arxiv.org/abs/2408.11048>

* **Problem.** RoboPianist needs a human to annotate fingering for every song. We do not
  have that for Runaway.
* **Fix: optimal-transport (OT) fingering.** At each step, solve an assignment problem:
  which finger should press which target key so the *total* fingertip-to-key distance is
  minimal (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`). Reward fingertips
  for being close to their assigned keys. The agent ends up *discovering* a fingering.
* **Result.** Matches human-fingering F1 on the songs they compared. They used it to
  train 2,089 per-song agents (most with F1 > 0.75).
* **Hyperparameters** (Table 4): 8M steps, batch 256, update every 2 steps, lookahead 10,
  gravity compensation on, control 0.05 s, **discount 0.88**, MLP 3×256 GELU, 2 Qs,
  dropout 0.01, lr 3e-4. ~21 h per song on an AMD MI250X.
* **MJX experiment (Appendix B.4).** They ported the environment to MuJoCo XLA (GPU,
  1024 parallel envs, ~159k steps/s after removing most contacts and tuning the solver)
  and trained PPO: 40M steps in 2 h for Twinkle Twinkle, but **lower F1 than DroQ**. So
  they went back to CPU + DroQ. Nobody has published FastTD3/FastSAC-style massively
  parallel *off-policy* training on the piano task yet: an open opportunity.

### Prior project: Humanoid Pianist (anandmaj, 2025)
Thread: <https://x.com/Almondgodd/status/1907584647602778521> · Code:
<https://github.com/AlmondGod/humanoid-pianist>

* Trained per-song **SAC** policies on RoboPianist for anime openings (10–15 s clips).
  Tried GRPO for robotics ("too slow") and visual RL ("too unstable") first.
* **Humanoid = cosmetic.** A Unitree G1 model is added to the scene and its arms follow
  the Shadow Hands through **inverse kinematics** (the `mink` library) every step, with
  a scripted "dance" for the torso/legs. RL only controls the hands.
* Fine-tuned with **CPL** (Contrastive Preference Learning, RLHF without RL): rate
  recorded performances, train the actor to prefer the liked ones, constrained close to
  the SAC policy. Reported as unstable.
* Hyperparameters that worked for him without fingering: lr 3e-4, batch 256, tau 0.01,
  initial temperature 1.2, discount 0.8.
* His clips were short because each policy only plays one short segment.

### HandelBot (Xie, Qi, Sadigh, 2026) — sim to real
Paper: <https://arxiv.org/abs/2603.12243>
* PPO per song in **ManiSkill** (GPU-parallel), then real-robot fine-tuning with residual
  RL from the keyboard's MIDI output (30 min of real data). Songs of 16–33 s.
* Restricted to 3 fingers per hand on hardware. Shows the "sim policy as a starting
  point" pattern if this ever goes onto a real robot.

### Newer work worth knowing (2026)
* **Adversarial Posture Regularization** (Qiu et al., arXiv:2606.23848): an AMP-style
  discriminator on hand poses, trained from casual Quest-3 hand-tracking data, makes
  hand shapes look human. Relevant later for making our robot *look* good.
  Code: <https://github.com/APRProject/APRPianist>
* **Expressive Robotic Pianist** (arXiv:2609.10844): adds musical dynamics (key velocity
  → loudness) and a human-hand-geometry imitation reward. Sim F1 ≈ 0.96.

### Simulation / compute options considered
| Option | Pros | Cons |
|---|---|---|
| RoboPianist on CPU MuJoCo (chosen first) | exact published env, known to work | ~300–800 steps/s per CPU core |
| MJX (MuJoCo on GPU, JAX) | 100k+ steps/s | contact-heavy scene is slow; RP1M's PPO underperformed |
| MuJoCo Warp / MuJoCo Playground | modern GPU MuJoCo, better contacts | no piano env exists yet: needs a port |
| ManiSkill / Genesis | GPU parallel | different models; rendering needs GPU graphics on SF Compute |
| Isaac Sim | — | does not run on H100 (no RT cores), per SF Compute docs |

## Decisions this led to

1. **Simulator: RoboPianist (MuJoCo, CPU)** for the first song. It is the environment
   every published result uses, so if our numbers are bad, the bug is ours.
2. **Reward: RoboPianist reward + RP1M's OT fingering**, because Runaway has no fingering
   annotations and the papers show learning fails without some fingering signal.
3. **Algorithm: SAC with DroQ-style critics** (dropout 0.01 + LayerNorm), discount 0.88,
   3×256 GELU networks: the recipe both RoboPianist and RP1M validated.
4. **Speed: many CPU environments in parallel per GPU** (one per core), with the replay
   buffer and many gradient steps per call on the GPU. Not GPU physics yet; a GPU
   port (MuJoCo Warp + a FastSAC-style learner) is a later research step, once we have a
   known-good baseline to compare against.
5. **Embodiment: hands first, humanoid second.** Phase 2 mounts the hands on
   a Unitree G1 like Humanoid Pianist did; Phase 3 could put the G1 arm joints under RL control.
6. **Evaluation: F1** like the papers, plus listening. F1 is computed per 50 ms step;
   an agent that presses nothing already scores 0.17 on our song, because the silent
   gaps between strikes count as "correct".
