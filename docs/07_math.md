# 07 · The math, step by step

Every formula the project relies on, with what it means and where it lives in the code.
Notation: vectors are bold-free for readability; $t$ is the control step (50 ms).

---

## 1. The problem as a Markov decision process

Playing the song is a **Markov decision process** (MDP) $(\mathcal S, \mathcal A, P, r, \gamma)$:

* **State** $s_t$: everything the policy is shown at step $t$ (hand joint angles, key
  positions, the piano roll for the next 0.5 s). A 1,114-dimensional vector
  (1,118 for the humanoid; 1,076 before the round-7 fix that restored the finger joint angles).
* **Action** $a_t \in [-1, 1]^{d}$: target joint positions, rescaled to each actuator's range
  ($d = 39$ for hands, $43$ for the humanoid with joint control, $41$ with hand-position (IK) control).
* **Dynamics** $P(s_{t+1} \mid s_t, a_t)$: MuJoCo, 10 physics steps of 5 ms per action.
* **Reward** $r(s_t, a_t)$: Section 3.
* **Discount** $\gamma \in [0, 1)$: how much a reward one step later is worth.

The goal is a policy $\pi(a \mid s)$ maximising the expected discounted return

$$
J(\pi) = \mathbb E_{\pi}\Big[\sum_{t=0}^{T-1} \gamma^t\, r(s_t, a_t)\Big],
\qquad T = 1{,}209 \text{ steps (60.4 s)}.
$$

**Why a small discount (0.84–0.92).** The *effective horizon* is about $1/(1-\gamma)$
steps: $\gamma = 0.88 \Rightarrow 8.3$ steps $\approx 0.42$ s. The policy is mostly judged
on the next half second, which matches the 0.5 s look-ahead it observes and the 1.38 s
spacing of strikes. With $\gamma = 0.99$ (100 steps, 5 s) value estimates mix many
future notes together; RoboPianist found such agents get cautious and skip notes.

---

## 2. From MIDI to a goal at every step

The song (Section 2 of [02_song_transcription.md](02_song_transcription.md)) is a list
of notes $(k_i, t^{\text{on}}_i, t^{\text{off}}_i)$: key, start, end. Strike $n$ starts at

$$
t^{\text{on}}_n = n \cdot \Delta, \qquad \Delta = \frac{2 \cdot 60}{87} = 1.379\,\text{s},
\qquad t^{\text{off}}_n = t^{\text{on}}_n + 0.8\,\Delta .
$$

Sampled every $\delta = 0.05$ s, this gives a binary **goal matrix**
$G \in \{0,1\}^{T \times 88}$ with $G_{t,k} = 1$ iff key $k$ should be down at step $t$.
The observation includes rows $t, t+1, \dots, t+10$ of $G$.

---

## 3. The reward

Per step, $r_t = r^{\text{key}}_t + r^{\text{OT}}_t + r^{\text{forearm}}_t + r^{\text{sustain}}_t + r^{\text{energy}}_t$.

### 3.1 The tolerance function
Most terms use dm_control's **tolerance** $g$: full reward inside a band, smooth Gaussian
decay outside it. For distance $x$, bound $b$ and margin $m$:

$$
g(x; b, m) = \begin{cases} 1 & x \le b \\[2pt]
\exp\!\Big(-\tfrac12\,\big(\tfrac{x - b}{m}\, c\big)^2\Big) & x > b \end{cases},
\qquad c = \sqrt{-2 \ln 0.1},
$$

so that $g(b + m) = 0.1$: at one margin past the band, the reward has fallen to 10%.

### 3.2 Key press
Let $K_t = \{k : G_{t,k} = 1\}$ be the keys that should be down, $q_k \in [0,1]$ how far key
$k$ is pressed (normalised joint angle), and $\text{on}_k$ whether it is sounding
(within 0.5° of fully down). Then

$$
r^{\text{key}}_t = \tfrac12 \cdot \frac{1}{|K_t|}\sum_{k \in K_t} g\big(1 - q_k;\ 0.05,\ 0.5\big)
\;+\; \tfrac12 \cdot \mathbb 1\big[\nexists\, k \notin K_t : \text{on}_k\big].
$$

The first half rewards pressing the right keys; the second half is lost entirely if **any**
wrong key sounds. That all-or-nothing penalty is what stops hands from hovering lazily on
top of the keys (a RoboPianist ablation).

### 3.3 Automatic fingering (optimal transport, RP1M)
Fingertip positions $f_1, \dots, f_{10} \in \mathbb R^3$, target points $p_k$ on each key in
$K_t$. Choose an assignment $w_{jk} \in \{0, 1\}$ (finger $j$ presses key $k$) solving

$$
\min_{w} \sum_{j, k} w_{jk}\,\lVert f_j - p_k \rVert_2
\quad\text{s.t.}\quad \sum_j w_{jk} = 1\ \ \forall k \in K_t,
\qquad \sum_k w_{jk} \le 1\ \ \forall j .
$$

Every key gets exactly one finger, and no finger takes two keys. This is a linear
assignment problem, solved exactly by the Hungarian / Jonker–Volgenant algorithm
(`scipy.optimize.linear_sum_assignment`, $O(n^3)$ with $n \le 10$: microseconds). With the
optimal pairs $(j^\*, k)$:

$$
r^{\text{OT}}_t = \frac{1}{|K_t|}\sum_{k \in K_t} g\big(\lVert f_{j^\*(k)} - p_k \rVert;\ 0.01,\ 0.1\big),
\qquad r^{\text{OT}}_t = 1 \text{ if } K_t = \varnothing .
$$

Because it is re-solved every step from the *current* hand pose, the cheapest finger
changes as the hand moves, and the policy settles on its own fingering.
Code: `RunawayPianoTask._compute_ot_fingering_reward` in `src/runaway/env.py`.

### 3.4 Anti-jitter terms (round 3, off by default)

**Extra-press penalty.** Let $\mathcal P_t$ be the keys that went down at step $t$
($A_{t,k} = 1$, $A_{t-1,k} = 0$). A press is *wanted* if its key's current note started at
most $w = 3$ steps ago (150 ms) and that note has not been struck yet. With
$E_t = |\mathcal P_t| - |\text{wanted presses}_t|$,

$$
r^{\text{press}}_t = -c\, E_t, \qquad c \in \{0.5, 2\}.
$$

Unlike $r^{\text{key}}$, which loses the same 0.5 whether one wrong key sounds for 50 ms
or ten keys do, this charges every audible extra note once, which is what the
note-level score measures.

**Arm command filter and rate penalty (G1 only).** The policy's arm action $a^{\text{arm}}_t$
is low-pass filtered before it reaches the servos, and fast changes cost reward:

$$
u_t = s\, u_{t-1} + (1 - s)\, a^{\text{arm}}_t, \qquad
r^{\text{rate}}_t = -\lambda\, \frac{1}{8}\sum_{j=1}^{8} \Big(\frac{u_{t,j} - u_{t-1,j}}{h_j}\Big)^2,
$$

with $s = 0.5$ (a time constant of about one control step), $h_j$ the half-range of joint
$j$, and $\lambda \in \{1, 5\}$. The filter alone removes the high-frequency part of the
command; the penalty teaches the policy not to ask for it.

### 3.5 The small terms
* $r^{\text{forearm}}_t = 0.5$ if the two forearms do not touch, else 0.
* $r^{\text{sustain}}_t = g(\text{pedal}_{\text{target}} - \text{pedal};\ 0.05, 0.5)$; the
  target is always "up" for our song.
* $r^{\text{energy}}_t = -0.005 \sum_i |\tau_i|\,|\dot q_i|$: actuator power (torque × joint
  speed), so the hands do not thrash.

---

## 4. The score: per-step precision, recall, F1

Let $A \in \{0,1\}^{T \times 88}$ be the keys actually sounding. At each step,

$$
\text{TP}_t = \sum_k G_{t,k} A_{t,k},\quad \text{FP}_t = \sum_k (1 - G_{t,k}) A_{t,k},\quad
\text{FN}_t = \sum_k G_{t,k} (1 - A_{t,k}),
$$

$$
P_t = \frac{\text{TP}_t}{\text{TP}_t + \text{FP}_t},\qquad
R_t = \frac{\text{TP}_t}{\text{TP}_t + \text{FN}_t},\qquad
F1_t = \frac{2\,\text{TP}_t}{2\,\text{TP}_t + \text{FP}_t + \text{FN}_t},
$$

with $0/0 := 1$. The reported score is the mean over steps, $\overline{F1} = \frac1T\sum_t F1_t$.

**Why doing nothing scores 0.175.** If $A = 0$, then $F1_t = 1$ on every silent step
($\text{TP} = \text{FP} = \text{FN} = 0$) and $0$ on every note step. Our clip has a key
held 80% of each 1.38 s slot, plus a 1 s lead-in and the gaps, and works out to 17.5% silent
steps. Code: `per_step_precision_recall_f1` (`env.py`), tested equal to scikit-learn.

### 4.1 Note-level score (how it sounds)

Per-step F1 forgives short blips: a finger that taps a neighbouring key for one 50 ms step
costs one false-positive step out of 1,209, yet it is an audible wrong note. So we also
score **onsets** (key-down events). Let $\mathcal O^\star = \{(t, k)\}$ be the target onsets
(rising edges of $G$) and $\mathcal O$ the robot's (rising edges of $A$). Match each target
$(t,k)$ to at most one unused robot onset $(t', k)$ on the same key with $|t' - t| \le 2$
steps (100 ms); let $h$ be the number of matches. Then

$$
P_{\text{note}} = \frac{h}{|\mathcal O|},\qquad R_{\text{note}} = \frac{h}{|\mathcal O^\star|},\qquad
F1_{\text{note}} = \frac{2 P_{\text{note}} R_{\text{note}}}{P_{\text{note}} + R_{\text{note}}} .
$$

Example (the G1 at 5.5M): $h = 40$, $|\mathcal O^\star| = 44$, $|\mathcal O| = 253$, so
$R = 0.91$ but $P = 0.16$ and $F1_{\text{note}} = 0.27$, even though per-step F1 is 0.876.
Code: `note_metrics` in `src/runaway/evaluate.py`.

---

## 5. Soft Actor-Critic (SAC)

SAC maximises reward **plus entropy**, so the policy stays as random as it can afford:

$$
J_{\text{SAC}}(\pi) = \mathbb E\Big[\sum_t \gamma^t \big(r_t + \alpha\, \mathcal H(\pi(\cdot \mid s_t))\big)\Big],
\qquad \mathcal H(\pi(\cdot\mid s)) = -\mathbb E_{a \sim \pi}[\log \pi(a \mid s)] .
$$

$\alpha > 0$ is the **temperature**: the price of being predictable.

### 5.1 The policy: a squashed Gaussian
The actor network outputs $\mu_\theta(s)$ and $\log \sigma_\theta(s)$ (clipped to $[-20, 2]$).

$$
u = \mu_\theta(s) + \sigma_\theta(s) \odot \varepsilon,\quad \varepsilon \sim \mathcal N(0, I),
\qquad a = \tanh(u) \in (-1, 1)^d .
$$

The *reparameterisation* ($a$ is a differentiable function of $\theta$ and independent noise
$\varepsilon$) lets gradients flow through sampling. Change of variables gives the
log-density of the squashed action:

$$
\log \pi_\theta(a \mid s) = \sum_{i=1}^d \Big[\log \mathcal N(u_i; \mu_i, \sigma_i)
- \log\big(1 - \tanh^2 u_i\big)\Big],
$$

and the code uses the numerically stable identity
$\log(1 - \tanh^2 u) = 2\,(\log 2 - u - \operatorname{softplus}(-2u))$
(`sample_tanh_gaussian` in `sac.py`).

### 5.2 The critics: soft Bellman regression
Two Q-networks $Q_{\phi_1}, Q_{\phi_2}$ and slow copies $Q_{\bar\phi_1}, Q_{\bar\phi_2}$. For a
replayed transition $(s, a, r, s', d)$ ($d = 0$ only on a true termination):

$$
y = r + \gamma\, d \Big(\min_{i=1,2} Q_{\bar\phi_i}(s', a') - \alpha \log \pi_\theta(a' \mid s')\Big),
\qquad a' \sim \pi_\theta(\cdot \mid s'),
$$

$$
\mathcal L_Q(\phi_i) = \mathbb E\big[(Q_{\phi_i}(s, a) - y)^2\big].
$$

The **minimum** of two critics counters the tendency of Q-learning to overestimate (any
noise in a max/argmax is biased upward). The target copies trail the live critics by
Polyak averaging, $\bar\phi \leftarrow (1 - \tau)\,\bar\phi + \tau\,\phi$ with $\tau = 0.005$,
which keeps the regression target from chasing itself.

**Song end ≠ failure.** When the song ends we keep $d = 1$ (a time limit, not a terminal
state), so the target still bootstraps. Setting $d = 0$ would teach the critic that the
world ends at 60 s.

### 5.3 DroQ: dropout + LayerNorm in the critic
Each critic layer is Dense → Dropout($p = 0.01$) → LayerNorm → GELU. A tiny dropout makes
each critic a cheap ensemble (random sub-networks), and LayerNorm bounds activations. Both
reduce over-fitting to the replay buffer, which is what lets RoboPianist (and us) take one
gradient step per environment step without the values diverging.

### 5.4 The actor: maximise soft value

$$
\mathcal L_\pi(\theta) = \mathbb E_{s \sim \mathcal D,\ \varepsilon}\Big[\alpha \log \pi_\theta(a_\theta(s, \varepsilon) \mid s)
- \tfrac12 \textstyle\sum_i Q_{\phi_i}(s, a_\theta(s, \varepsilon))\Big].
$$

Minimising it pushes actions toward higher Q while the $\alpha \log \pi$ term rewards
spread (high entropy).

### 5.5 Automatic temperature
Pick a target entropy $\bar{\mathcal H} = -\tfrac12 d$ and adjust $\alpha$ by gradient descent on

$$
\mathcal L(\alpha) = \alpha\,\big(\mathcal H_{\text{current}} - \bar{\mathcal H}\big) .
$$

If the policy is less random than the target, the gradient increases $\alpha$, making
entropy more valuable, and vice versa. We optimise $\log \alpha$ so that $\alpha$ stays positive.

### 5.6 Update-to-data ratio
With $N$ parallel environments each loop adds $N$ transitions and runs
$\lceil \text{UTD} \cdot N \rceil$ gradient steps, each on a fresh uniform minibatch of 256
from a 1M-transition buffer. UTD = 1 matches RoboPianist (one step per transition). The
"batch 1024, UTD 0.25" variant takes 4× fewer, 4× bigger steps: the same number of sampled
transitions per environment step, but far fewer GPU kernel launches.

---

## 6. The humanoid's kinematics, briefly

On RoboPianist's sliders a hand's position is two numbers the policy sets directly. On the
G1 it is the end of a kinematic chain. The palm position is

$$
x_{\text{palm}} = T_{\text{base}}\; R_y(q_1) R_x(q_2) R_z(q_3)\, T_{\text{upper}}\; R_y(q_4)\, T_{\text{fore}}\, x_0 ,
$$

(shoulder pitch, roll, yaw, then elbow). A small joint change moves the hand by
$\Delta x \approx J(q)\,\Delta q$, where the Jacobian $J$ scales with arm length (~0.4 m). So an
exploration step of 0.1 rad at the shoulder moves the fingertips ~4 cm, more than the
1 cm "close enough" band of the fingering reward. This is why the arm control range
(±0.4 rad) and the piano height mattered so much (06_results.md, round 2): too wide a range
and random exploration rarely brings a finger near a key, so the reward never fires.

### 6.1 Hand-position control through inverse kinematics (round 4)

With `arm_control="ik"` the policy outputs, per hand, a wrist offset
$o \in [-1, 1]^3$; the target is $x^\star = x_{\text{home}} + o \odot (0.08, 0.30, 0.08)$ m
(toward/away, along the keyboard, up/down). Each control step a few iterations of
**damped least squares** move the four arm joints $q$ toward it (on a scratch copy of the
state; the robot only receives the result as servo targets):

$$
\Delta q = J^\top (J J^\top + \mu^2 I)^{-1} (x^\star - x(q))
\;+\; \big(I - J^\top (J J^\top + \mu^2 I)^{-1} J\big)\, k\,(q_{\text{home}} - q),
$$

with $J = \partial x / \partial q \in \mathbb R^{3 \times 4}$ (MuJoCo's `mj_jacBody`),
$\mu = 0.05$ and $k = 0.1$. The damping keeps steps finite near singular poses; the second
term uses the arm's one spare degree of freedom to stay close to a natural posture without
disturbing the wrist. From the policy's point of view the wrist now behaves like
RoboPianist's sliders (smooth, bounded, directly in task space), but the motion is produced
by the G1's real shoulder and elbow joints. Measured tracking with the hand lifted: 0.1–0.3
cm once settled; an 18 cm move settles in ≈ 0.5 s.

Each arm joint is a position servo with gravity compensation:
$\tau = k_p (q^{\text{target}} - q) - k_d \dot q$, with $k_p \in \{80, 120\}$ N·m/rad.

---

## 7. Throughput arithmetic

One loop iteration with $N = 20$ environments took (8×H100 node, per run):

$$
t_{\text{iter}} = t_{\text{act}} + t_{\text{launch}} + t_{\text{sim}} + t_{\text{store}}
= 3.0 + 1.8 + 7.2 + 0.3 = 12.3\ \text{ms}
\;\Rightarrow\; \frac{20}{12.3\ \text{ms}} \approx 1{,}600\ \text{steps/s}.
$$

GPU updates run concurrently with the simulation, so they only add their *launch* time.
3M steps at 1,500 steps/s take 33 min. 8 runs on one node cost
$\$40.32/\text{h} \times 0.55\,\text{h} \approx \$22$, i.e. **≈ $2.80 per 3M-step run**.
