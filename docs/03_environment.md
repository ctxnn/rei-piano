# 03 · The environment: what the robot sees, does, and is rewarded for

Code: [`src/runaway/env.py`](../src/runaway/env.py) (task + wrappers),
[`src/runaway/vec_env.py`](../src/runaway/vec_env.py) (parallel copies).

## The scene

![](../outputs/frames/smoke_frame.png)

* **Piano:** RoboPianist's full 88-key model. Each key is a hinge with a spring; a key
  "sounds" when it is within 0.5° of fully down. Keys light up green when pressed in our
  videos, and keys the song wants right now are shown in blue.
* **Hands:** two Shadow Dexterous Hands (MuJoCo Menagerie). Each forearm slides along
  the keyboard on a prismatic joint (`forearm_tx`) and can move up/down (`forearm_ty`),
  so the policy moves the whole hand as well as the fingers.
* **Timing:** physics every 5 ms; the policy acts every 50 ms (20 Hz). One performance of
  the 60 s clip = **1,209 control steps** (1 s lead-in + 60.4 s of music).

## Observation (1,114 numbers)

| Part | Size | Meaning |
|---|---|---|
| `goal` | 11 × 89 = 979 | piano roll for now + the next 10 steps (0.5 s): which of 88 keys (+ pedal) should be down |
| `piano/state` | 88 | how far each key is pressed (0–1) |
| `piano/sustain_state` | 1 | pedal |
| `rh_shadow_hand/joints_pos` | 23 | right hand joint angles + forearm position |
| `lh_shadow_hand/joints_pos` | 23 | left hand |

There is **no fingering observation**: RoboPianist normally tells the agent which finger
to use; we do not have that for Runaway.

## Action (39 numbers in [−1, 1])

Target joint positions for both hands (19 each with RoboPianist's *reduced action space*,
which removes a few rarely-useful wrist/finger DoFs and limits the thumb) + 1 sustain
pedal. MuJoCo position actuators track the targets. Gravity compensation is on, so a hand
holds its pose without fighting gravity.

## Reward per step

| Term | Range | What it encourages |
|---|---|---|
| key press | 0–1 | 0.5 × "the right keys are down" + 0.5 × "no wrong key sounds" |
| OT fingering (ours, from RP1M) | 0–1 | fingertips close to the keys that should be down |
| forearm | 0 or 0.5 | forearms not colliding |
| sustain | 0–1 | pedal in the right state (always "up" for us) |
| energy | ≤ 0 | −0.005 × actuator power: calm motion |

**OT fingering in plain words.** At every step, take the keys that should be down and
all 10 fingertips. Find the assignment of fingers to keys (each finger at most one key)
that makes the total distance smallest: that is the Hungarian algorithm,
`scipy.optimize.linear_sum_assignment`. Reward those fingertips for being within ~1 cm
of their key (smooth Gaussian falloff to ~0 at 10 cm). Because it is recomputed every
step from where the hands actually are, the robot invents its own fingering.

## Metric

Precision/recall/F1 of "keys down" vs "keys that should be down", computed per control
step and averaged over the song (RoboPianist's definition, zero-division counted as 1).
**Baseline: doing nothing scores F1 = 0.175** on our clip, because during the 0.28 s gaps
between strikes "no key down" is correct. Anything meaningfully above 0.2 is real playing.

## Knobs (`EnvConfig`)

| Setting | Default | Why |
|---|---|---|
| `midi_path` | `data/midi/runaway_60s.mid` | 45 s version also available |
| `n_steps_lookahead` | 10 | RoboPianist/RP1M value (0.5 s of future notes) |
| `initial_buffer_time` | 1.0 s | right hand starts 13 cm left of the first note (E6) |
| `gravity_compensation`, `reduced_action_space`, `primitive_fingertip_collisions` | on | the paper's best settings |
| `ot_fingering` | on | without any fingering signal, learning stalls (paper ablation) |
| `wrong_press_termination` | off | end the episode on a wrong note; a later fine-tuning option |

## Speed notes

* One environment steps at ~830 steps/s on an Apple M5 core and ~260–290 on a Xeon
  8480+ core (a cloud H100 node).
* RoboPianist's evaluation wrapper computed F1 with scikit-learn, once per step at the end
  of every episode: 42% of all simulation time. `FastMidiEvaluationWrapper` does the same
  computation in NumPy (identical results, `tests/test_metrics.py`).
