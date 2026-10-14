"""Soft Actor-Critic (SAC) in JAX, set up the way RoboPianist trains its pianists.

SAC in one paragraph
--------------------
An *actor* network maps an observation to a Gaussian over actions (squashed by tanh
into [-1, 1]). Two *critic* networks estimate Q(s, a): the discounted sum of future
rewards after taking action a in state s. The critics learn from replayed experience
with the Bellman target  r + gamma * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s')),
where Q' are slowly-updated copies of the critics. The actor learns to pick actions
the critics like while keeping its entropy high ("try many things"); alpha, the
temperature, is tuned automatically so the entropy stays near a target.

RoboPianist-specific choices (from the paper and its code)
---------------------------------------------------------
* Critics use LayerNorm and 1% dropout ("DroQ"), which keeps Q-values stable.
* Discount ~0.8-0.9, much lower than the usual 0.99: piano notes need precise
  timing *now*, and a long horizon makes agents cautious (they skip notes).
* 3 hidden layers of 256 units, GELU activations.

Performance design
------------------
The replay buffer lives on the accelerator (GPU) as JAX arrays, and `update` runs
many gradient steps inside a single compiled `lax.scan`, so there is no Python or
host-to-device overhead per gradient step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass(frozen=True)  # frozen: JAX needs it hashable (it is static in jit).
class SACConfig:
    hidden_dims: Sequence[int] = (256, 256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    discount: float = 0.88  # RP1M used 0.88; the RoboPianist paper found 0.84-0.92 best.
    tau: float = 0.005  # Target-critic smoothing: Q' <- (1 - tau) Q' + tau Q.
    init_temperature: float = 1.0
    target_entropy_per_dim: float = -0.5  # target entropy = this * action_dim
    critic_dropout: float = 0.01
    critic_layer_norm: bool = True
    num_qs: int = 2
    batch_size: int = 256


# ----------------------------------------------------------------------------- networks


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    dropout: float = 0.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for size in self.hidden_dims:
            x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)
            if self.dropout > 0:
                x = nn.Dropout(self.dropout, deterministic=not train)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        return x


class Actor(nn.Module):
    """Outputs the mean and log-std of a Gaussian; actions are tanh(sample)."""

    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = MLP(self.hidden_dims)(obs)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(h)
        log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(h)
        return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


class NumpyActor:
    """A CPU copy of the actor, for choosing actions without a GPU round trip.

    Asking the GPU for 12 actions costs ~4 ms of host time per step (copy in, launch,
    copy out) while the network itself is tiny: 3 layers of 256 units. Computing it in
    NumPy takes ~0.2 ms. The learner keeps training on the GPU; every few iterations
    `load()` copies the latest weights over. A slightly stale actor is fine for SAC,
    which learns off-policy from whatever data the replay buffer holds.

    Must match `Actor` above exactly (flax's default GELU is the tanh approximation).
    """

    def __init__(self, params) -> None:
        self.load(params)

    def load(self, params) -> None:
        p = jax.device_get(params)
        mlp = p["MLP_0"]
        self.hidden = [(mlp[f"Dense_{i}"]["kernel"], mlp[f"Dense_{i}"]["bias"]) for i in range(len(mlp))]
        self.mean_layer = (p["Dense_0"]["kernel"], p["Dense_0"]["bias"])
        self.log_std_layer = (p["Dense_1"]["kernel"], p["Dense_1"]["bias"])

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def __call__(self, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Stochastic actions if `rng` is given, else the deterministic (mean) action."""
        h = obs.astype(np.float32)
        for w, b in self.hidden:
            h = self._gelu(h @ w + b)
        mean = h @ self.mean_layer[0] + self.mean_layer[1]
        if rng is None:
            return np.tanh(mean)
        log_std = np.clip(h @ self.log_std_layer[0] + self.log_std_layer[1], LOG_STD_MIN, LOG_STD_MAX)
        return np.tanh(mean + np.exp(log_std) * rng.standard_normal(mean.shape)).astype(np.float32)


class Critic(nn.Module):
    """`num_qs` independent Q-networks, evaluated in parallel with vmap."""

    hidden_dims: Sequence[int]
    dropout: float
    layer_norm: bool
    num_qs: int

    @nn.compact
    def __call__(self, obs, action, train: bool = False) -> jnp.ndarray:
        ensemble = nn.vmap(
            _SingleQ,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return ensemble(self.hidden_dims, self.dropout, self.layer_norm)(obs, action, train)


class _SingleQ(nn.Module):
    hidden_dims: Sequence[int]
    dropout: float
    layer_norm: bool

    @nn.compact
    def __call__(self, obs, action, train: bool) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(self.hidden_dims, self.dropout, self.layer_norm)(x, train)
        return nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x).squeeze(-1)


def sample_tanh_gaussian(key, mean, log_std):
    """Sample a = tanh(u), u ~ N(mean, std), and return log pi(a) with the tanh correction."""
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    log_prob = jax.scipy.stats.norm.logpdf(u, mean, std).sum(-1)
    # d tanh(u)/du = 1 - tanh(u)^2; this numerically stable form is from the SAC paper code.
    log_prob -= (2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))).sum(-1)
    return jnp.tanh(u), log_prob


# ----------------------------------------------------------------------------- replay


class ReplayBuffer(struct.PyTreeNode):
    """Fixed-size ring buffer of transitions, stored on the accelerator."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray  # 0 if the episode truly terminated, else 1.
    next_obs: jnp.ndarray
    ptr: jnp.ndarray  # next write position
    size: jnp.ndarray  # number of valid entries

    @staticmethod
    def create(capacity: int, obs_dim: int, action_dim: int) -> "ReplayBuffer":
        return ReplayBuffer(
            obs=jnp.zeros((capacity, obs_dim), jnp.float32),
            action=jnp.zeros((capacity, action_dim), jnp.float32),
            reward=jnp.zeros((capacity,), jnp.float32),
            discount=jnp.zeros((capacity,), jnp.float32),
            next_obs=jnp.zeros((capacity, obs_dim), jnp.float32),
            ptr=jnp.zeros((), jnp.int32),
            size=jnp.zeros((), jnp.int32),
        )

    @property
    def capacity(self) -> int:
        return self.reward.shape[0]

    def sample(self, key, batch_size: int) -> dict:
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        return dict(
            obs=self.obs[idx], action=self.action[idx], reward=self.reward[idx],
            discount=self.discount[idx], next_obs=self.next_obs[idx],
        )  # fmt: skip


def pack_transitions(obs, action, reward, discount, next_obs) -> np.ndarray:
    """Put a batch of transitions side by side in one float32 array (host side).

    Copying one array to the GPU costs one transfer latency (~0.6 ms on our nodes);
    copying five small arrays costs five. See `add_packed_to_buffer`.
    """
    return np.concatenate(
        [obs, action, reward[:, None], discount[:, None], next_obs], axis=1, dtype=np.float32
    )


@partial(jax.jit, donate_argnums=0)
def add_packed_to_buffer(buf: ReplayBuffer, packed: jnp.ndarray) -> ReplayBuffer:
    """Unpack (on the GPU) an array made by `pack_transitions` and insert it."""
    obs_dim, act_dim = buf.obs.shape[1], buf.action.shape[1]
    obs = packed[:, :obs_dim]
    action = packed[:, obs_dim : obs_dim + act_dim]
    reward = packed[:, obs_dim + act_dim]
    discount = packed[:, obs_dim + act_dim + 1]
    next_obs = packed[:, obs_dim + act_dim + 2 :]
    return add_to_buffer(buf, obs, action, reward, discount, next_obs)  # inlined by jit


@partial(jax.jit, donate_argnums=0)
def add_to_buffer(buf: ReplayBuffer, obs, action, reward, discount, next_obs) -> ReplayBuffer:
    """Insert a batch of N transitions (one per parallel environment)."""
    n = obs.shape[0]
    idx = (buf.ptr + jnp.arange(n)) % buf.capacity
    return buf.replace(
        obs=buf.obs.at[idx].set(obs),
        action=buf.action.at[idx].set(action),
        reward=buf.reward.at[idx].set(reward),
        discount=buf.discount.at[idx].set(discount),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        ptr=(buf.ptr + n) % buf.capacity,
        size=jnp.minimum(buf.size + n, buf.capacity),
    )


# ----------------------------------------------------------------------------- agent


class SAC(struct.PyTreeNode):
    actor: TrainState
    critic: TrainState
    target_critic_params: dict
    log_temp: TrainState
    rng: jnp.ndarray
    cfg: SACConfig = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)

    @staticmethod
    def create(cfg: SACConfig, obs_dim: int, action_dim: int, seed: int = 0) -> "SAC":
        rng = jax.random.PRNGKey(seed)
        rng, k_actor, k_critic = jax.random.split(rng, 3)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, action_dim))

        actor_def = Actor(cfg.hidden_dims, action_dim)
        critic_def = Critic(cfg.hidden_dims, cfg.critic_dropout, cfg.critic_layer_norm, cfg.num_qs)
        critic_params = critic_def.init(k_critic, obs, act)["params"]
        log_temp = {"log_temp": jnp.log(jnp.asarray(cfg.init_temperature, jnp.float32))}
        return SAC(
            actor=TrainState.create(
                apply_fn=actor_def.apply,
                params=actor_def.init(k_actor, obs)["params"],
                tx=optax.adam(cfg.actor_lr),
            ),
            critic=TrainState.create(
                apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(cfg.critic_lr)
            ),
            target_critic_params=critic_params,
            log_temp=TrainState.create(
                apply_fn=None, params=log_temp, tx=optax.adam(cfg.temp_lr)
            ),
            rng=rng,
            cfg=cfg,
            target_entropy=cfg.target_entropy_per_dim * action_dim,
        )

    # --- acting

    @jax.jit
    def sample_actions(self, obs: jnp.ndarray) -> tuple["SAC", jnp.ndarray]:
        """Stochastic actions, used while collecting training data."""
        rng, key = jax.random.split(self.rng)
        mean, log_std = self.actor.apply_fn({"params": self.actor.params}, obs)
        action, _ = sample_tanh_gaussian(key, mean, log_std)
        return self.replace(rng=rng), action

    @jax.jit
    def eval_actions(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic actions (the Gaussian's mean), used for evaluation."""
        mean, _ = self.actor.apply_fn({"params": self.actor.params}, obs)
        return jnp.tanh(mean)

    # --- learning

    def _update_once(self, batch: dict) -> tuple["SAC", dict]:
        cfg = self.cfg
        rng, k_next, k_drop1, k_pi, k_drop2 = jax.random.split(self.rng, 5)
        temp = jnp.exp(self.log_temp.params["log_temp"])

        # 1) Critic: regress Q(s, a) toward the soft Bellman target.
        mean, log_std = self.actor.apply_fn({"params": self.actor.params}, batch["next_obs"])
        next_a, next_logp = sample_tanh_gaussian(k_next, mean, log_std)
        next_q = self.critic.apply_fn(
            {"params": self.target_critic_params}, batch["next_obs"], next_a, True,
            rngs={"dropout": k_drop1},
        ).min(axis=0)  # fmt: skip
        target = batch["reward"] + cfg.discount * batch["discount"] * (next_q - temp * next_logp)
        target = jax.lax.stop_gradient(target)

        def critic_loss_fn(params):
            qs = self.critic.apply_fn(
                {"params": params}, batch["obs"], batch["action"], True, rngs={"dropout": k_drop1}
            )
            return ((qs - target) ** 2).mean(), qs.mean()

        (critic_loss, q_mean), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            self.critic.params
        )
        critic = self.critic.apply_gradients(grads=grads)
        target_params = optax.incremental_update(critic.params, self.target_critic_params, cfg.tau)

        # 2) Actor: maximise  Q(s, pi(s)) - temp * log pi(a|s).
        def actor_loss_fn(params):
            mean, log_std = self.actor.apply_fn({"params": params}, batch["obs"])
            a, logp = sample_tanh_gaussian(k_pi, mean, log_std)
            q = critic.apply_fn(
                {"params": critic.params}, batch["obs"], a, True, rngs={"dropout": k_drop2}
            ).mean(axis=0)
            return (temp * logp - q).mean(), -logp.mean()

        (actor_loss, entropy), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            self.actor.params
        )
        actor = self.actor.apply_gradients(grads=grads)

        # 3) Temperature: raise it if entropy is below target, lower it if above.
        def temp_loss_fn(params):
            return -(jnp.exp(params["log_temp"]) * (self.target_entropy - entropy))

        log_temp = self.log_temp.apply_gradients(grads=jax.grad(temp_loss_fn)(self.log_temp.params))

        info = dict(critic_loss=critic_loss, actor_loss=actor_loss, q_mean=q_mean,
                    entropy=entropy, temperature=temp)  # fmt: skip
        new = self.replace(actor=actor, critic=critic, target_critic_params=target_params,
                           log_temp=log_temp, rng=rng)  # fmt: skip
        return new, info

    @partial(jax.jit, static_argnames="num_updates")
    def update(self, buffer: ReplayBuffer, num_updates: int) -> tuple["SAC", dict]:
        """Run `num_updates` gradient steps, each on a fresh random minibatch."""

        def step(agent: SAC, _):
            rng, key = jax.random.split(agent.rng)
            batch = buffer.sample(key, agent.cfg.batch_size)
            return agent.replace(rng=rng)._update_once(batch)

        # Measured on an H100 node with 12 updates per call: rolled scan ~4 ms of host
        # time, unroll=True ~14 ms (more kernels to launch) plus ~70 s compile. The cost
        # is the CPU launching hundreds of small GPU kernels, so keep it rolled.
        agent, infos = jax.lax.scan(step, self, None, length=num_updates)
        return agent, jax.tree.map(lambda x: x.mean(), infos)


def actor_params_to_numpy(agent: SAC) -> dict:
    return jax.tree.map(np.asarray, agent.actor.params)
