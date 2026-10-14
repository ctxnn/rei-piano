"""The CPU (NumPy) copy of the actor must compute the same actions as the JAX actor."""

import numpy as np

from runaway.sac import SAC, NumpyActor, SACConfig


def test_numpy_actor_matches_jax_actor():
    agent = SAC.create(SACConfig(), obs_dim=1114, action_dim=39, seed=3)
    obs = np.random.default_rng(0).standard_normal((12, 1114)).astype(np.float32)
    jax_actions = np.asarray(agent.eval_actions(obs))
    np_actions = NumpyActor(agent.actor.params)(obs)
    np.testing.assert_allclose(np_actions, jax_actions, atol=1e-5)
