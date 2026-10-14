"""Phase 2: a Unitree G1 humanoid playing the piano with Shadow Hands.

In RoboPianist, each hand hangs on two invisible sliders (forearm_tx / forearm_ty) that
move it along and toward the keyboard. Here those sliders are gone. Each Shadow Hand is
bolted to the wrist of a Unitree G1 humanoid, and the policy must move the G1's
**shoulders and elbows** to bring the hands to the notes, the way a person does.

    G1 (fixed pelvis, legs and waist locked)
     └─ shoulder pitch / roll / yaw, elbow      <- 4 RL-controlled joints per arm
         └─ Shadow Hand at the wrist            <- wrist + finger joints, RL-controlled
              (the Shadow's own forearm housing is hidden: the G1 forearm is the forearm)

Everything else (piano, song, reward, metrics) is the same task as the hands-only phase.
The action vector becomes [right hand, left hand, right arm (4), left arm (4), pedal].

Model source: MuJoCo Menagerie `unitree_g1/g1.xml` (BSD-3), vendored by
`scripts/get_g1.sh` into third_party/mujoco_menagerie.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import physics_utils, spec_utils
from robopianist.models.hands import HandSide, shadow_hand

from runaway.env import RunawayPianoTask

ROOT = Path(__file__).resolve().parents[2]
G1_XML = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")

# Scene layout (meters). RoboPianist's piano sits on the floor; a standing G1 needs it
# at table height. The G1 stands behind the keyboard (+x side), facing it (-x).
PIANO_HEIGHT = 0.75  # fingertips ~4 cm above the keys at the home pose
G1_POSITION = (0.45, 0.0, 0.0)
G1_YAW = np.pi  # G1 faces +x by default; turn it around to face the piano.

# The Shadow forearm housing is 0.213 m from its base to the wrist joint. We place its
# base so that the Shadow wrist lands where the G1's own wrist was.
_SHADOW_FOREARM_LENGTH = 0.213
_G1_WRIST_FROM_ELBOW = np.array([0.10 + 0.038 + 0.046, 0.0, -0.01])

def _hand_mount_quat() -> np.ndarray:
    """Orientation of a Shadow Hand mount in the G1 elbow frame.

    RoboPianist attaches its hands to the world with quaternion (-1, -1, 1, 1)/2, which
    makes them point at the keys with palms down. We want the same world orientation
    when the G1 arm is at its zero pose, where the elbow frame is the world turned 180
    degrees about z (the G1 faces the piano). So mount = R_elbow^T @ R_robopianist.
    (Evaluates to (0.5, 0.5, 0.5, 0.5); verified numerically against the built model.)
    """
    r_world = np.zeros(9)
    mujoco.mju_quat2Mat(r_world, np.array([-1.0, -1.0, 1.0, 1.0]) / 2)
    r_elbow = np.diag([-1.0, -1.0, 1.0])  # 180 degrees about z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (r_elbow.T @ r_world.reshape(3, 3)).reshape(-1))
    return quat


_HAND_MOUNT_QUAT = _hand_mount_quat()

# Position-control gains for the arm joints (N*m/rad), and how far from the home pose
# the policy may command them (rad).
#
# The range matters a lot. With +-1.2 rad, random (exploring) arm commands threw the
# hands anywhere within ~40 cm: fingertips were near the key surface only 12% of the
# time (vs 55% for RoboPianist's sliders) and 8 runs learned nothing in 300k steps.
# +-0.4 rad with the piano at 0.75 m gives 55% and still lets the right hand reach
# y = -0.10..0.43 m (the song needs 0.12..0.32). Measured with random actions
# (docs/06_results.md, round 2).
ARM_KP = {"shoulder_pitch": 120.0, "shoulder_roll": 120.0, "shoulder_yaw": 80.0, "elbow": 80.0}
ARM_CTRL_HALF_RANGE = 0.4
# Home pose: forearms forward and slightly down, hands just above the keys.
ARM_HOME = {"shoulder_pitch": -0.35, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": 0.55}

# --- Hand-position (IK) control, `arm_control="ik"` ---------------------------------
# Instead of 4 joint targets per arm, the policy outputs where each wrist should be,
# as an offset from its home position: (toward/away from the keys, along the keyboard,
# up/down), each in [-1, 1] and scaled by these half-ranges (meters). That is what
# RoboPianist's sliders give its hands, so the hands move just as steadily, but here the
# G1's shoulders and elbows really carry them (inverse kinematics, every control step).
#
# The vertical range matters most. RoboPianist's slider cannot lower a hand far enough to
# press keys, so its policies press with individual fingers. With +-12 cm (round 5) the
# G1 could push the whole hand into the keyboard (4-6 keys at once) and learned to play
# by dabbing: 571 presses for 44 strikes, 54% one-step blips. With +-3 cm the lowest
# wrist leaves the fingertips just above the keys: only finger motion presses keys.
IK_TARGET_HALF_RANGE = np.array([0.03, 0.30, 0.03])  # see the comment above
# The joints then may use most of their range (the bounded Cartesian target keeps
# exploration sane, which is what the +-0.4 rad joint limit did in joint mode).
IK_JOINT_HALF_RANGE = 1.2
IK_ITERATIONS = 4
IK_DAMPING = 0.05  # damped least squares: stable near singular arm poses
IK_NULLSPACE_GAIN = 0.1  # pull the spare degree of freedom toward the home posture


def _look_at_xyaxes(eye, target, up=(0, 0, 1)) -> list[float]:
    """MuJoCo camera orientation (its x and y axes) for a camera at `eye` facing `target`."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return [*right, *cam_up]


# Named cameras for videos: (eye, target).
CAMERAS = {
    "front": ((-1.25, 0.0, 1.45), (0.35, 0.0, 0.85)),       # across the piano, at the G1
    "front_left": ((-0.9, 1.1, 1.35), (0.35, 0.0, 0.85)),
    "over_shoulder": ((1.25, -0.55, 1.55), (0.0, 0.1, 0.70)),  # behind the G1, at the keys
}


class G1(composer.Entity):
    """A Unitree G1 with fixed pelvis, locked legs/waist, and position-controlled arms."""

    def _build(self, xml_path: Path = G1_XML, ctrl_half_range: float = ARM_CTRL_HALF_RANGE,
               stiffness_scale: float = 1.0) -> None:  # fmt: skip
        self._mjcf_root = mjcf.from_path(str(xml_path))
        self._mjcf_root.model = "g1"
        # Fixed base: remove the free joint so the robot stands still.
        for joint in list(self._mjcf_root.find_all("joint")):
            if joint.tag == "freejoint" or joint.type == "free":
                joint.remove()
        # Remove the G1's own hands: everything from the wrist onward.
        for side in ("left", "right"):
            self._mjcf_root.find("body", f"{side}_wrist_roll_link").remove()
        # Lock every joint except the arm joints we control (legs, waist).
        keep = {f"{s}_{j}_joint" for s in ("left", "right") for j in ARM_JOINTS}
        for joint in list(self._mjcf_root.find_all("joint")):
            if joint.name not in keep:
                joint.remove()
        # Replace Menagerie's actuators with our own position actuators.
        for act in list(self._mjcf_root.find_all("actuator")):
            act.remove()
        for key in list(self._mjcf_root.find_all("key")):
            key.remove()  # keyframes refer to the joints we removed
        self._arm_joints = {}
        self._arm_actuators = {}
        for side in ("right", "left"):
            joints, acts = [], []
            for j in ARM_JOINTS:
                joint = self._mjcf_root.find("joint", f"{side}_{j}_joint")
                # Servo stiffness kp and joint damping scale together (critical-ish damping).
                # Round 1-4 used scale 1: a 5 N key press then pushed the hand ~6 mm
                # sideways (docs/06_results.md, round 4 root cause), enough to hit the next key.
                joint.damping = 2.0 * np.sqrt(stiffness_scale)
                joint.armature = 0.01
                home = ARM_HOME[j] * (-1 if (side == "left" and j in ("shoulder_roll", "shoulder_yaw")) else 1)
                lo, hi = joint.range
                ctrl = (max(lo, home - ctrl_half_range), min(hi, home + ctrl_half_range))
                acts.append(self._mjcf_root.actuator.add(
                    "position", name=f"{side}_{j}", joint=joint, kp=ARM_KP[j] * stiffness_scale,
                    ctrlrange=ctrl, forcerange=(-25 * max(1.0, stiffness_scale ** 0.5), 25 * max(1.0, stiffness_scale ** 0.5)),
                ))  # fmt: skip
                joints.append(joint)
            self._arm_joints[side] = joints
            self._arm_actuators[side] = acts
        # A mount point for each Shadow Hand where the G1 wrist used to start.
        self._hand_mounts = {}
        for side in ("right", "left"):
            elbow = self._mjcf_root.find("body", f"{side}_elbow_link")
            sign = -1 if side == "right" else 1
            pos = _G1_WRIST_FROM_ELBOW - np.array([_SHADOW_FOREARM_LENGTH, 0, 0])
            pos[1] = sign * 0.0019
            self._hand_mounts[side] = elbow.add(
                "site", name=f"{side}_hand_mount", pos=pos, quat=_HAND_MOUNT_QUAT
            )
        physics_utils.compensate_gravity(self._mjcf_root)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def arm_joints(self, side: str):
        return self._arm_joints[side]

    def arm_actuators(self, side: str):
        return self._arm_actuators[side]

    def hand_mount(self, side: str):
        return self._hand_mounts[side]

    @staticmethod
    def home_angle(side: str, joint: str) -> float:
        return ARM_HOME[joint] * (-1 if (side == "left" and joint in ("shoulder_roll", "shoulder_yaw")) else 1)

    def set_home_pose(self, physics) -> None:
        for side in ("right", "left"):
            for joint, act, j in zip(self._arm_joints[side], self._arm_actuators[side], ARM_JOINTS):
                home = ARM_HOME[j] * (-1 if (side == "left" and j in ("shoulder_roll", "shoulder_yaw")) else 1)
                physics.bind(joint).qpos = home
                physics.bind(act).ctrl = home


class HumanoidPianoTask(RunawayPianoTask):
    """The Runaway task, played by a G1 humanoid whose arms carry the Shadow Hands.

    Arm control modes (EnvConfig.arm_control):
      "joint"  the policy sets 4 joint targets per arm (shoulder pitch/roll/yaw, elbow)
      "ik"     the policy sets a wrist position per hand; inverse kinematics turns it into
               the 4 joint targets (see IK_* constants above)
    Optional anti-jitter measures in both modes: a low-pass filter on the arm commands
    (joint targets or wrist targets), and a reward penalty on how fast they change.
    """

    def __init__(self, arm_action_smoothing: float = 0.0, arm_action_rate_penalty: float = 0.0,
                 arm_control: str = "joint", arm_stiffness_scale: float = 1.0,
                 piano_height: float = PIANO_HEIGHT, **kwargs) -> None:  # fmt: skip
        if not 0.0 <= arm_action_smoothing < 1.0:
            raise ValueError("arm_action_smoothing must be in [0, 1)")
        if arm_control not in ("joint", "ik"):
            raise ValueError(f"arm_control must be 'joint' or 'ik', got {arm_control!r}")
        self._arm_smoothing = arm_action_smoothing
        self._arm_rate_penalty = arm_action_rate_penalty
        self._arm_control = arm_control
        self._arm_stiffness_scale = arm_stiffness_scale
        self._piano_height = piano_height
        self._ik_data = None
        super().__init__(**kwargs)

    def _set_rewards(self) -> None:
        super()._set_rewards()
        if self._arm_rate_penalty > 0:
            self._reward_fn.add("arm_action_rate_penalty", self._compute_arm_rate_penalty)

    def _add_hand(self, hand_side, position, quaternion, gravity_compensation,
                  primitive_fingertip_collisions, reduced_action_space, attachment_yaw,
                  forearm_dofs):  # fmt: skip
        del position, quaternion, attachment_yaw, forearm_dofs  # set by the G1 instead
        if not hasattr(self, "_g1"):
            self._g1 = G1(ctrl_half_range=IK_JOINT_HALF_RANGE if self._arm_control == "ik"
                          else ARM_CTRL_HALF_RANGE,
                          stiffness_scale=self._arm_stiffness_scale)  # fmt: skip
            frame = self._arena.attach(self._g1)
            frame.pos = G1_POSITION
            frame.euler = (0, 0, G1_YAW)
            mjcf.get_attachment_frame(self._piano.mjcf_model).pos = (0, 0, self._piano_height)
            for name, (eye, target) in CAMERAS.items():
                self._arena.mjcf_model.worldbody.add(
                    "camera", name=f"humanoid_{name}", pos=eye, xyaxes=_look_at_xyaxes(eye, target), fovy=45
                )
        hand = shadow_hand.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=(),  # no sliders: the G1 arm moves the hand
        )
        # RoboPianist *replaces* the forearm body's built-in rotation with its mount
        # rotation; do the same, so the mount site alone sets the hand's orientation.
        hand.root_body.quat = (1, 0, 0, 0)
        # Hide the Shadow forearm housing and stop it colliding; the G1 forearm is the arm.
        forearm = hand.mjcf_model.find("body", hand.root_body.name)
        for geom in forearm.find_all("geom", immediate_children_only=True):
            geom.rgba = (0, 0, 0, 0)
            geom.contype = geom.conaffinity = 0
        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)
        side = "right" if hand_side == HandSide.RIGHT else "left"
        # Attach through composer (Entity.attach), not the raw MJCF site: only then is the
        # hand a child entity whose observables (finger joint angles) reach the policy.
        # Rounds 2-6 used `site.attach(hand.mjcf_model)`: physics and actions worked, but
        # the policy never saw its own fingers (observation had only the 8 arm joints).
        self._g1.attach(hand, attach_site=self._g1.hand_mount(side))
        return hand

    @property
    def g1(self) -> G1:
        return self._g1

    def initialize_episode(self, physics, random_state) -> None:
        super().initialize_episode(physics, random_state)
        self._g1.set_home_pose(physics)
        acts = self._g1.arm_actuators("right") + self._g1.arm_actuators("left")
        ranges = np.array([physics.bind(a).ctrlrange for a in acts])
        self._arm_center = ranges.mean(axis=1)
        self._arm_half_range = (ranges[:, 1] - ranges[:, 0]) / 2
        self._arm_ctrl = np.asarray(physics.bind(acts).ctrl).copy()  # filtered command
        self._arm_rate = 0.0
        if self._arm_control == "ik":
            self._init_ik(physics)

    # --- inverse kinematics -----------------------------------------------------------

    def _init_ik(self, physics) -> None:
        """Index the arm joints and wrist bodies; remember the home wrist positions."""
        model = physics.model
        physics.forward()  # update body positions for the home pose just set
        self._ik = {}
        for side, hand in (("right", self.right_hand), ("left", self.left_hand)):
            joint_ids = [physics.bind(j).element_id for j in self._g1.arm_joints(side)]
            wrist = hand.mjcf_model.find("body", hand.mjcf_model.model.split("_")[0] + "_wrist")
            body_id = physics.bind(wrist).element_id
            self._ik[side] = dict(
                body_id=body_id,
                qpos_adr=np.array([model.jnt_qposadr[j] for j in joint_ids]),
                dof_adr=np.array([model.jnt_dofadr[j] for j in joint_ids]),
                home_q=np.array([G1.home_angle(side, j) for j in ARM_JOINTS]),
                lo=np.array([physics.bind(a).ctrlrange[0] for a in self._g1.arm_actuators(side)]),
                hi=np.array([physics.bind(a).ctrlrange[1] for a in self._g1.arm_actuators(side)]),
                home_pos=physics.data.xpos[body_id].copy(),
            )
        if self._ik_data is None:
            self._ik_data = mujoco.MjData(model.ptr)
        self._ik_targets = np.concatenate([self._ik["right"]["home_pos"], self._ik["left"]["home_pos"]])
        self._arm_ctrl = np.zeros(6)  # filtered normalised wrist offsets

    def _solve_ik(self, physics, targets: dict) -> dict:
        """Joint angles that put each wrist at its target (damped least squares).

        Starts from the current configuration, so consecutive solutions are close and the
        arm moves smoothly. Works on a scratch copy of the state: the simulation itself is
        only touched through the position servos' targets.
        """
        model, data = physics.model.ptr, self._ik_data
        data.qpos[:] = physics.data.qpos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        eye3 = np.eye(3)
        for _ in range(IK_ITERATIONS):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            for side, ik in self._ik.items():
                err = targets[side] - data.xpos[ik["body_id"]]
                mujoco.mj_jacBody(model, data, jacp, jacr, ik["body_id"])
                jac = jacp[:, ik["dof_adr"]]  # 3 x 4
                q = data.qpos[ik["qpos_adr"]]
                inv = np.linalg.solve(jac @ jac.T + IK_DAMPING**2 * eye3, np.eye(3))
                dq = jac.T @ inv @ err
                nullspace = np.eye(4) - jac.T @ inv @ jac
                dq += nullspace @ (IK_NULLSPACE_GAIN * (ik["home_q"] - q))
                data.qpos[ik["qpos_adr"]] = np.clip(q + dq, ik["lo"], ik["hi"])
        return {side: data.qpos[ik["qpos_adr"]].copy() for side, ik in self._ik.items()}

    def _compute_arm_rate_penalty(self, physics) -> float:
        del physics  # Unused.
        return -self._arm_rate_penalty * self._arm_rate

    def before_step(self, physics, action, random_state) -> None:
        n_hand = self.right_hand.action_spec(physics).shape[0]
        rh, lh = action[:n_hand], action[n_hand : 2 * n_hand]
        self.right_hand.apply_action(physics, rh, random_state)
        self.left_hand.apply_action(physics, lh, random_state)
        if self._arm_control == "ik":
            self._apply_ik_action(physics, np.asarray(action[2 * n_hand : 2 * n_hand + 6], np.float64))
            self.piano.apply_sustain(physics, action[-1], random_state)
            return
        arm = np.asarray(action[2 * n_hand : 2 * n_hand + 8], dtype=np.float64)
        # Squared change of the commanded arm targets, in units of each joint's half range.
        previous = self._arm_ctrl
        smoothed = self._arm_smoothing * previous + (1 - self._arm_smoothing) * arm
        self._arm_rate = float(np.mean(((smoothed - previous) / self._arm_half_range) ** 2))
        self._arm_ctrl = smoothed
        physics.bind(self._g1.arm_actuators("right")).ctrl = smoothed[:4]
        physics.bind(self._g1.arm_actuators("left")).ctrl = smoothed[4:]
        self.piano.apply_sustain(physics, action[-1], random_state)

    def _apply_ik_action(self, physics, offsets: np.ndarray) -> None:
        """offsets: 6 numbers in [-1, 1], (x, y, z) for the right then the left wrist."""
        previous = self._arm_ctrl
        smoothed = self._arm_smoothing * previous + (1 - self._arm_smoothing) * offsets
        self._arm_rate = float(np.mean((smoothed - previous) ** 2))
        self._arm_ctrl = smoothed
        targets = {
            "right": self._ik["right"]["home_pos"] + smoothed[:3] * IK_TARGET_HALF_RANGE,
            "left": self._ik["left"]["home_pos"] + smoothed[3:] * IK_TARGET_HALF_RANGE,
        }
        joints = self._solve_ik(physics, targets)
        physics.bind(self._g1.arm_actuators("right")).ctrl = joints["right"]
        physics.bind(self._g1.arm_actuators("left")).ctrl = joints["left"]

    def action_spec(self, physics) -> specs.BoundedArray:
        def arm_spec(side):
            if self._arm_control == "ik":
                return specs.BoundedArray((3,), np.float64, -np.ones(3), np.ones(3), name=f"{side}_wrist")
            ranges = np.array([physics.bind(a).ctrlrange for a in self._g1.arm_actuators(side)])
            return specs.BoundedArray((4,), np.float64, ranges[:, 0], ranges[:, 1], name=f"{side}_arm")

        sustain = specs.BoundedArray((1,), np.float64, [0.0], [1.0], name="sustain")
        return spec_utils.merge_specs([
            self.right_hand.action_spec(physics), self.left_hand.action_spec(physics),
            arm_spec("right"), arm_spec("left"), sustain,
        ])  # fmt: skip

    def _add_observables(self) -> None:
        super()._add_observables()

        def arm_joints(physics):
            joints = self._g1.arm_joints("right") + self._g1.arm_joints("left")
            return np.asarray(physics.bind(joints).qpos)

        obs = observable.Generic(arm_joints)
        obs.enabled = True
        self._task_observables["g1_arm_joints"] = obs
