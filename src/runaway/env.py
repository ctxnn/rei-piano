"""The piano-playing environment: two Shadow Hands, an 88-key piano, and our song.

We build on RoboPianist (Zakka et al., 2023): it provides the MuJoCo models of the
piano and the hands and a task that turns a MIDI file into a reward. We add one thing,
the *automatic fingering* reward from RP1M (Zhao et al., 2024), because RoboPianist's
dense shaping reward needs a human to annotate which finger presses which key, and
nobody has annotated Runaway.

One environment step = 0.05 s of music (20 Hz control, 10 physics steps of 5 ms).

What the policy sees (observation, 1 vector of ~1100 floats):
  * goal:       which of the 88 keys (+ sustain pedal) should be down now and for the
                next `n_steps_lookahead` steps (a scrolling piano roll)
  * piano state: how far each key is pressed right now
  * joints:      joint angles of both hands, including the forearm sliders that move
                 each hand along the keyboard

What the policy does (action, 39 numbers in [-1, 1]):
  target joint angles for both hands (19 each with the reduced action space) and the
  sustain pedal. MuJoCo position actuators then drive the joints toward the targets.

Reward per step (each term is in roughly [0, 1]):
  key_press   0.5 * (are the right keys down?) + 0.5 * (no wrong key is sounding)
  fingering   are fingertips close to the keys that should be pressed? (see below)
  forearm     0.5 if the two forearms are not touching
  sustain     is the pedal in the right state? (always "up" for our song)
  energy      small penalty on actuator power, so motions stay calm
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dm_env
import numpy as np
from dm_control import composer
from dm_control.utils.rewards import tolerance
from dm_env_wrappers import CanonicalSpecWrapper, ConcatObservationWrapper, EnvironmentWrapper
from robopianist.music import midi_file
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from robopianist.wrappers.evaluation import EpisodeMetrics, MidiEvaluationWrapper
from scipy.optimize import linear_sum_assignment

# Same constants as RoboPianist's own fingering reward: full reward once a fingertip is
# within 1 cm of its key, decaying smoothly to ~0 at 10 cm.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01


@dataclass
class EnvConfig:
    midi_path: str = "data/midi/runaway_60s.mid"
    # How many future control steps of the piano roll the policy sees (10 = 0.5 s).
    n_steps_lookahead: int = 10
    control_timestep: float = 0.05
    # Seconds of silence before the first note, so the hand can travel to it.
    initial_buffer_time: float = 1.0
    # Standard RoboPianist simplifications (all on in the paper's best runs):
    gravity_compensation: bool = True  # hands do not sag under their own weight
    reduced_action_space: bool = True  # drop a few hard-to-use joints, limit thumb
    primitive_fingertip_collisions: bool = True  # capsule fingertips: faster contacts
    disable_forearm_reward: bool = False
    energy_penalty_coef: float = 5e-3
    # Automatic fingering (RP1M). Off = no dense shaping at all (learning stalls).
    ot_fingering: bool = True
    # End the episode as soon as a wrong key sounds. Useful late in training only.
    wrong_press_termination: bool = False
    # Who plays: "hands" = two Shadow Hands on sliders (RoboPianist); "g1" = the hands
    # mounted on a Unitree G1 humanoid whose arm joints the policy controls
    # (see humanoid.py).
    embodiment: str = "hands"
    # --- Fixes for a jittery performance (added after the G1 round; 0 = off) ---
    # Reward lost for every key press that does not start a wanted note: a brushed
    # neighbour, a re-press of a held key, a note at the wrong time. RoboPianist's key
    # reward only loses 0.5 *while* any wrong key is down, so a 50 ms tap is nearly free.
    extra_press_penalty: float = 0.0
    # A press counts as starting a wanted note if it comes within this many control steps
    # after the note's start (3 = 150 ms) and is the first press of that note.
    onset_window_steps: int = 3
    # G1 only: low-pass filter on arm commands, ctrl = s * previous + (1 - s) * action.
    arm_action_smoothing: float = 0.0
    # G1 only: reward lost per unit of squared change in the (normalised) arm command.
    arm_action_rate_penalty: float = 0.0
    # G1 only: "joint" = the policy sets shoulder/elbow targets; "ik" = the policy sets
    # wrist positions and inverse kinematics moves the arm (humanoid.py).
    arm_control: str = "joint"
    # G1 only: multiply the arm servos' stiffness (kp); damping scales with its square
    # root and the torque limit too. 1 = rounds 2-4 (a key press shoves the hand ~6 mm
    # sideways); higher = a stiffer arm, closer to RoboPianist's sliders.
    arm_stiffness_scale: float = 1.0
    # G1 only: keyboard height (m). 0.75 (rounds 2-4) puts the fingertips ON the keys at
    # the neutral pose: resting fingers creep sideways and brush keys (the G1's extra
    # presses). 0.70 makes the hands hover ~9 cm above, like RoboPianist's (use with IK,
    # whose wrist z range reaches the keys).
    piano_height: float = 0.75


class RunawayPianoTask(PianoWithShadowHands):
    """RoboPianist's two-hand task plus the RP1M automatic-fingering reward."""

    def __init__(self, ot_fingering: bool = True, extra_press_penalty: float = 0.0,
                 onset_window_steps: int = 3, **kwargs) -> None:  # fmt: skip
        self._ot_fingering = ot_fingering
        self._extra_press_penalty = extra_press_penalty
        self._onset_window_steps = onset_window_steps
        super().__init__(**kwargs)

    def _set_rewards(self) -> None:
        super()._set_rewards()
        # With no fingering annotations, the parent class has no shaping reward that
        # pulls fingers toward keys. Add ours.
        if self._disable_fingering_reward and self._ot_fingering:
            self._reward_fn.add("ot_fingering_reward", self._compute_ot_fingering_reward)
        if self._extra_press_penalty > 0:
            self._reward_fn.add("extra_press_penalty", self._compute_extra_press_penalty)

    # --- Extra-press penalty: counts key-down events that do not start a wanted note.

    def initialize_episode(self, physics, random_state) -> None:
        super().initialize_episode(physics, random_state)
        n = self.piano.n_keys
        self._prev_keys_down = np.zeros(n, bool)
        self._prev_goal = np.zeros(n, bool)
        self._steps_since_note_start = np.full(n, 10**6)
        self._note_already_struck = np.ones(n, bool)
        self._extra_presses = 0

    def after_step(self, physics, random_state) -> None:
        super().after_step(physics, random_state)
        goal = self._goal_current[:-1].astype(bool)
        keys_down = self.piano.activation.copy()
        # Bookkeeping for each key: when did its current note start, and has it been struck?
        note_start = goal & ~self._prev_goal
        self._steps_since_note_start = np.where(note_start, 0, self._steps_since_note_start + 1)
        self._note_already_struck[note_start] = False
        presses = keys_down & ~self._prev_keys_down
        wanted = (presses & goal & ~self._note_already_struck
                  & (self._steps_since_note_start <= self._onset_window_steps))  # fmt: skip
        self._note_already_struck |= wanted
        self._extra_presses = int(presses.sum() - wanted.sum())
        self._prev_keys_down, self._prev_goal = keys_down, goal

    def _compute_extra_press_penalty(self, physics) -> float:
        del physics  # Unused.
        return -self._extra_press_penalty * self._extra_presses

    def _key_target_position(self, physics, key: int) -> np.ndarray:
        """Where a fingertip should touch a key: top surface, a bit toward the front.

        Same point RoboPianist uses for its annotated-fingering reward.
        """
        geom = physics.bind(self.piano.keys[key].geom[0])
        pos = geom.xpos.copy()
        pos[2] += 0.5 * geom.size[2]
        pos[0] += 0.35 * geom.size[0]
        return pos

    def _compute_ot_fingering_reward(self, physics) -> float:
        """Automatic fingering as an assignment problem (RP1M, arXiv:2408.11048).

        For the keys that should be down right now, pick which finger presses which
        key so that the *total* fingertip-to-key distance is smallest (each finger
        takes at most one key). Reward is high when those assigned fingertips are on
        their keys. Recomputed every step, so the "fingering" adapts as the hands move,
        and the agent effectively discovers its own fingering.
        """
        keys = np.flatnonzero(self._goal_current[:-1])
        if keys.size == 0:
            return 1.0  # Nothing to press: nothing to be close to.
        fingertips = np.array(
            [physics.bind(site).xpos for hand in (self.left_hand, self.right_hand)
             for site in hand.fingertip_sites]
        )  # fmt: skip  # (10, 3)
        targets = np.array([self._key_target_position(physics, k) for k in keys])  # (K, 3)
        dist = np.linalg.norm(fingertips[:, None, :] - targets[None, :, :], axis=-1)  # (10, K)
        rows, cols = linear_sum_assignment(dist)  # Hungarian algorithm, minimises sum.
        rewards = tolerance(
            dist[rows, cols],
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=_FINGER_CLOSE_ENOUGH_TO_KEY * 10,
            sigmoid="gaussian",
        )
        return float(np.mean(rewards))


class Float32ObservationWrapper(EnvironmentWrapper):
    """Cast observations and rewards to float32 (what the networks use).

    Replaces dm_env_wrappers.SinglePrecisionWrapper, which breaks on NumPy 2.
    """

    def _convert(self, ts: dm_env.TimeStep) -> dm_env.TimeStep:
        reward = None if ts.reward is None else np.float32(ts.reward)
        return ts._replace(observation=np.asarray(ts.observation, np.float32), reward=reward)

    def reset(self) -> dm_env.TimeStep:
        return self._convert(self._environment.reset())

    def step(self, action) -> dm_env.TimeStep:
        return self._convert(self._environment.step(action))


def per_step_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> EpisodeMetrics:
    """Musical metrics exactly as RoboPianist defines them, vectorised with NumPy.

    Both arrays are (T, K) booleans: for each of T control steps, which of K keys should
    be down / were down. Precision, recall and F1 are computed per step and averaged over
    steps. A step where nothing should be pressed and nothing was pressed counts as
    perfect (scikit-learn's `zero_division=1`).

    RoboPianist calls scikit-learn once per step, which took ~40% of all simulation
    time; this gives identical numbers (tested in tests/test_metrics.py) far faster.
    """
    t, p = y_true.astype(bool), y_pred.astype(bool)
    tp = (t & p).sum(axis=1)
    fp = (~t & p).sum(axis=1)
    fn = (t & ~p).sum(axis=1)

    def ratio(num, den):
        return np.where(den > 0, num / np.maximum(den, 1), 1.0)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return EpisodeMetrics(float(precision.mean()), float(recall.mean()), float(f1.mean()))


class FastMidiEvaluationWrapper(MidiEvaluationWrapper):
    """RoboPianist's MidiEvaluationWrapper with the per-step scikit-learn loop replaced."""

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        task = self._environment.task
        truth = np.zeros((len(task._notes), task.piano.n_keys), dtype=bool)
        for i, notes in enumerate(task._notes):
            truth[i, [n.key for n in notes]] = True
        pred = np.asarray(self._key_presses, dtype=bool)
        return per_step_precision_recall_f1(truth[: len(pred)], pred)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        task = self._environment.task
        truth = np.asarray([np.atleast_1d(v) for v in task._sustains], dtype=bool)
        pred = np.asarray(self._sustain_presses, dtype=bool)
        return per_step_precision_recall_f1(truth[: len(pred)], pred)


def make_env(cfg: EnvConfig, seed: int | None = None) -> dm_env.Environment:
    """A dm_env with flat float32 observations and actions scaled to [-1, 1].

    The returned env also tracks musical precision/recall/F1 per episode:
    `env.get_musical_metrics()` after an episode ends.
    """
    midi = midi_file.MidiFile.from_file(Path(cfg.midi_path))
    if cfg.embodiment == "hands":
        task_cls = RunawayPianoTask
    elif cfg.embodiment == "g1":
        from runaway.humanoid import HumanoidPianoTask as task_cls
    else:
        raise ValueError(f"unknown embodiment {cfg.embodiment!r}")
    extra = {}
    if cfg.embodiment == "g1":
        extra = dict(arm_action_smoothing=cfg.arm_action_smoothing,
                     arm_action_rate_penalty=cfg.arm_action_rate_penalty,
                     arm_control=cfg.arm_control,
                     arm_stiffness_scale=cfg.arm_stiffness_scale,
                     piano_height=cfg.piano_height)  # fmt: skip
    task = task_cls(
        **extra,
        ot_fingering=cfg.ot_fingering,
        extra_press_penalty=cfg.extra_press_penalty,
        onset_window_steps=cfg.onset_window_steps,
        midi=midi,
        n_steps_lookahead=cfg.n_steps_lookahead,
        initial_buffer_time=cfg.initial_buffer_time,
        wrong_press_termination=cfg.wrong_press_termination,
        disable_fingering_reward=True,  # No human fingering for Runaway.
        disable_forearm_reward=cfg.disable_forearm_reward,
        energy_penalty_coef=cfg.energy_penalty_coef,
        gravity_compensation=cfg.gravity_compensation,
        reduced_action_space=cfg.reduced_action_space,
        primitive_fingertip_collisions=cfg.primitive_fingertip_collisions,
        control_timestep=cfg.control_timestep,
        change_color_on_activation=True,  # Pressed keys light up in renders.
    )
    # Compile the model single-threaded. MuJoCo otherwise starts a thread per mesh while
    # compiling; with 8 runs x 20 workers starting at once on a node that allows 4,096
    # threads, compiles of the mesh-heavy G1 failed with "Caught an unknown exception".
    task.root_entity.mjcf_model.compiler.usethread = False
    # Room for HD offscreen renders (MuJoCo's default framebuffer is 640x480).
    visual_global = getattr(task.root_entity.mjcf_model.visual, "global")
    visual_global.offwidth, visual_global.offheight = 1920, 1080
    env = composer.Environment(
        task=task,
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
    )
    env = FastMidiEvaluationWrapper(env)
    env = ConcatObservationWrapper(env)
    env = CanonicalSpecWrapper(env, clip=True)
    env = Float32ObservationWrapper(env)
    return env


def episode_length(env: dm_env.Environment) -> int:
    """Number of control steps in one full performance of the song."""
    return len(env.task._notes)
