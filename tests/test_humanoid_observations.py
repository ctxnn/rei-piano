"""The G1 policy must observe its own finger joints (rounds 2-6 silently did not)."""

import warnings

from runaway.env import EnvConfig, make_env


def _raw_observation_spec(env):
    while hasattr(env, "_environment"):
        env = env._environment
    return env.observation_spec()


def test_g1_observes_hand_and_arm_joints():
    warnings.filterwarnings("ignore")
    for arm_control in ("joint", "ik"):
        env = make_env(EnvConfig(embodiment="g1", arm_control=arm_control, piano_height=0.70))
        env.reset()
        spec = _raw_observation_spec(env)
        hand_obs = [k for k in spec if k.endswith("shadow_hand/joints_pos")]
        assert len(hand_obs) == 2, f"finger joints missing from the observation: {list(spec)}"
        assert spec["g1_arm_joints"].shape == (8,)
