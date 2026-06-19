from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from slipzero.sensors import SlipDetector


@dataclass(frozen=True)
class Phase1Metrics:
    lifted_height: float
    min_hold_lift: float
    tilt_deg: float
    max_hold_tilt_deg: float
    fingers_engaged: int
    force_closure: bool


@dataclass(frozen=True)
class Phase1Result:
    passed: bool
    state: str
    reason: str
    metrics: Phase1Metrics


class Phase1FSM:
    def __init__(self, env, impedance, contacts, config):
        self.env = env
        self.impedance = impedance
        self.contacts = contacts
        self.config = config
        self.state = "APPROACH"
        self.reason = ""
        self.pick = np.asarray(config["pick_pos"], float)
        self.lift = self.pick + np.array([0.0, 0.0, float(config["lift_height"])])
        self.open_hand = np.full(env._hand_ctrl.shape, float(config["hand_open"]))
        self.close_hand = np.asarray(config["hand_close"], float)
        self.quat = env.ee_quat()
        self.start_vial_z = float(env.sensor("vial_pos")[2])
        self.min_hold_lift = float("inf")
        self.max_hold_tilt = 0.0
        self.hold_samples = 0
        self.state_steps = 0

    def run(self):
        while self.state not in {"DONE", "FAILED"}:
            self.step()
        metrics = self.metrics()
        return Phase1Result(self.state == "DONE", self.state, self.reason, metrics)

    def step(self):
        if self.state == "APPROACH":
            self._apply(self.pick, self.open_hand)
            if (
                self.state_steps >= self.config["approach_ready_steps"]
                and self._ee_error(self.pick) <= self.config["ee_tolerance"]
            ):
                self._transition("GRASP")
            elif self.state_steps >= self.config["approach_max_steps"]:
                self._fail("approach did not reach the pick pose")
        elif self.state == "GRASP":
            alpha = min(1.0, (self.state_steps + 1) / self.config["hand_close_steps"])
            self._apply(self.pick, self.close_hand * alpha)
            if self.state_steps >= self.config["grasp_ready_steps"] and self._has_grasp():
                self._transition("LIFT")
            if self.state_steps >= self.config["grasp_max_steps"]:
                self._fail("grasp did not reach force closure")
        elif self.state == "LIFT":
            self._apply(self.lift, self.close_hand)
            if (
                self._lifted_height() >= self.config["min_lift"]
                and self._tilt_deg() <= self.config["max_tilt_deg"]
                and self._has_grasp()
                and self._ee_error(self.lift) <= self.config["lift_ee_tolerance"]
                and self.state_steps >= self.config["lift_ready_steps"]
            ):
                self._transition("HOLD")
            elif self.state_steps >= self.config["lift_max_steps"]:
                self._fail("vial did not lift high enough")
        elif self.state == "HOLD":
            self._apply(self.lift, self.close_hand)
            lifted = self._lifted_height()
            tilt = self._tilt_deg()
            self.min_hold_lift = min(self.min_hold_lift, lifted)
            self.max_hold_tilt = max(self.max_hold_tilt, tilt)
            if lifted < self.config["min_lift"]:
                self._fail("vial dropped during hold")
            elif tilt > self.config["max_tilt_deg"]:
                self._fail("vial tilted during hold")
            elif self.contacts.fingers_engaged(self.config["f_grasp_min"]) < self.config["min_fingers"]:
                self._fail("grip was lost during hold")
            else:
                self.hold_samples += 1
                if self.hold_samples >= self.config["hold_steps"]:
                    self.reason = "held the lifted vial upright"
                    self.state = "DONE"
        self.state_steps += 1

    def metrics(self):
        hold_lift = self.min_hold_lift
        if not np.isfinite(hold_lift):
            hold_lift = self._lifted_height()
        return Phase1Metrics(
            lifted_height=self._lifted_height(),
            min_hold_lift=hold_lift,
            tilt_deg=self._tilt_deg(),
            max_hold_tilt_deg=self.max_hold_tilt,
            fingers_engaged=self.contacts.fingers_engaged(self.config["f_grasp_min"]),
            force_closure=self.contacts.has_force_closure(self.config["f_grasp_min"]),
        )

    def _apply(self, target, hand):
        self.env.data.ctrl[self.env._hand_ctrl] = hand
        self.impedance.apply(target, self.quat)
        self.env.step()

    def _transition(self, state):
        self.state = state
        self.state_steps = 0

    def _fail(self, reason):
        self.reason = reason
        self.state = "FAILED"

    def _ee_error(self, target):
        return float(np.linalg.norm(self.env.ee_pos() - target))

    def _lifted_height(self):
        return float(self.env.sensor("vial_pos")[2] - self.start_vial_z)

    def _has_grasp(self):
        f_min = self.config["f_grasp_min"]
        return (
            self.contacts.fingers_engaged(f_min) >= self.config["min_fingers"]
            and self.contacts.has_force_closure(f_min)
        )

    def _tilt_deg(self):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.env.sensor("vial_quat"))
        axis_alignment = abs(rotation.reshape(3, 3)[2, 2])
        return float(np.degrees(np.arccos(np.clip(axis_alignment, -1.0, 1.0))))


@dataclass(frozen=True)
class Phase2Metrics:
    slip_events: int
    recovery_latency_ms: float | None
    min_margin: float
    final_margin: float
    start_margin: float
    peak_grip_force: float
    final_lift: float
    min_lift: float
    final_tilt_deg: float
    max_tilt_deg: float
    max_displacement: float
    fingers_engaged: int
    force_closure: bool
    dropped: bool


@dataclass(frozen=True)
class Phase2Result:
    passed: bool
    reason: str
    metrics: Phase2Metrics


class Phase2FSM:
    def __init__(self, env, impedance, contacts, phase1_config, phase2_config, recovery_enabled=True):
        self.env = env
        self.impedance = impedance
        self.contacts = contacts
        self.phase1_config = phase1_config
        self.config = phase2_config
        self.recovery_enabled = recovery_enabled
        self.phase1 = Phase1FSM(env, impedance, contacts, phase1_config)
        self.detector = SlipDetector(
            contacts,
            phase2_config["margin_threshold"],
            debounce_steps=phase2_config["debounce_steps"],
            filter_alpha=phase2_config["filter_alpha"],
        )
        self.recovery_hand = np.asarray(phase2_config["recovery_hand"], float)
        self.base_hand = np.asarray(
            phase2_config.get(
                "base_hand",
                (self.phase1.close_hand * float(phase2_config.get("base_hand_scale", 1.0))).tolist(),
            ),
            float,
        )
        self.recovery_offset = np.asarray(phase2_config["recovery_target_offset"], float)
        self.impulse_force = np.asarray(phase2_config["impulse_force"], float)
        self.slip_events = 0
        self.first_slip_step = None
        self.recovery_step = None
        self.margin_recovered_step = None
        self.margin_trace = []
        self.min_margin = float("inf")
        self.peak_grip_force = 0.0
        self.min_lift = float("inf")
        self.max_tilt = 0.0
        self.max_displacement = 0.0
        self.start_margin = float("inf")
        self.start_vial_pos = None
        self._slipping = False

    def run(self):
        self._reach_phase1_hold()
        if self.phase1.state == "FAILED":
            return Phase2Result(False, self.phase1.reason, self.metrics())
        self.detector.reset()
        self.start_vial_pos = self.env.sensor("vial_pos")
        self.start_margin = self.contacts.min_friction_margin()
        for step in range(self.config["total_steps"]):
            self._step_trial(step)
        self.env.clear_vial_force()
        metrics = self.metrics()
        passed = (
            metrics.slip_events > 0
            and metrics.recovery_latency_ms is not None
            and not metrics.dropped
            and metrics.final_tilt_deg <= self.config["max_tilt_deg"]
            and metrics.final_margin >= self.config["margin_threshold"]
            and metrics.fingers_engaged >= self.config["min_fingers"]
            and metrics.force_closure
        )
        reason = "recovered from slip" if passed else "slip recovery failed"
        return Phase2Result(passed, reason, metrics)

    def metrics(self):
        final_margin = self.margin_trace[-1] if self.margin_trace else self.contacts.min_friction_margin()
        latency = None
        if (
            self.recovery_step is not None
            and self.first_slip_step is not None
            and self.margin_recovered_step is not None
        ):
            latency = (self.margin_recovered_step - self.first_slip_step) * self.env.model.opt.timestep * 1000.0
        min_lift = self.min_lift
        if not np.isfinite(min_lift):
            min_lift = self._lifted_height()
        return Phase2Metrics(
            slip_events=self.slip_events,
            recovery_latency_ms=latency,
            min_margin=self.min_margin,
            final_margin=final_margin,
            start_margin=self.start_margin,
            peak_grip_force=self.peak_grip_force,
            final_lift=self._lifted_height(),
            min_lift=min_lift,
            final_tilt_deg=self._tilt_deg(),
            max_tilt_deg=self.max_tilt,
            max_displacement=self.max_displacement,
            fingers_engaged=self.contacts.fingers_engaged(self.config["f_grasp_min"]),
            force_closure=self.contacts.has_force_closure(self.config["f_grasp_min"]),
            dropped=self._dropped(),
        )

    def _reach_phase1_hold(self):
        while self.phase1.state not in {"HOLD", "FAILED"}:
            self.phase1.step()

    def _step_trial(self, step):
        self._apply_impulse(step)
        recovering = self.recovery_enabled and self.recovery_step is not None
        hand = self.recovery_hand if recovering else self.base_hand
        target = self.phase1.lift + self.recovery_offset if recovering else self.phase1.lift
        self.env.data.ctrl[self.env._hand_ctrl] = hand
        self.impedance.apply(target, self.phase1.quat)
        self.env.step()
        margin = self.detector.update()
        self.margin_trace.append(margin)
        self.min_margin = min(self.min_margin, margin)
        forces = self.contacts.finger_normal_forces()
        self.peak_grip_force = max(self.peak_grip_force, float(np.max(forces)))
        self.min_lift = min(self.min_lift, self._lifted_height())
        self.max_tilt = max(self.max_tilt, self._tilt_deg())
        self.max_displacement = max(
            self.max_displacement,
            float(np.linalg.norm(self.env.sensor("vial_pos") - self.start_vial_pos)),
        )
        self._update_slip_state(step, margin)

    def _apply_impulse(self, step):
        start = self.config["impulse_start_steps"]
        end = start + self.config["impulse_steps"]
        if start <= step < end:
            self.env.apply_vial_force(self.impulse_force)
        else:
            self.env.clear_vial_force()

    def _update_slip_state(self, step, margin):
        if self.detector.slip_imminent and not self._slipping:
            if self.first_slip_step is None:
                self.slip_events = 1
                self.first_slip_step = step
            if self.recovery_enabled and self.recovery_step is None:
                self.recovery_step = step
            self._slipping = True
        elif not self.detector.slip_imminent:
            self._slipping = False
        if (
            self.first_slip_step is not None
            and self.margin_recovered_step is None
            and margin >= self.config["margin_threshold"]
        ):
            self.margin_recovered_step = step

    def _lifted_height(self):
        return float(self.env.sensor("vial_pos")[2] - self.phase1.start_vial_z)

    def _dropped(self):
        return self._lifted_height() < self.config["drop_lift"]

    def _tilt_deg(self):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.env.sensor("vial_quat"))
        axis_alignment = abs(rotation.reshape(3, 3)[2, 2])
        return float(np.degrees(np.arccos(np.clip(axis_alignment, -1.0, 1.0))))


@dataclass(frozen=True)
class Phase3Metrics:
    vial_in_port: bool
    ee_retracted: bool
    contaminated: bool
    uncapped: bool
    drops: int
    cap_angle_deg: float
    cap_delta_deg: float
    uncap_angle_deg: float
    port_distance: float
    final_vial_pos: tuple


@dataclass(frozen=True)
class Phase3Result:
    passed: bool
    state: str
    reason: str
    metrics: Phase3Metrics


class Phase3FSM:
    def __init__(self, env, impedance, contacts, phase1_config, phase3_config):
        self.env = env
        self.impedance = impedance
        self.contacts = contacts
        self.config = phase3_config
        self.phase1_config = phase1_config
        self.phase1 = Phase1FSM(env, impedance, contacts, phase1_config)
        self.uncap_hand = np.asarray(
            phase3_config.get("uncap_hand", phase1_config["hand_close"]),
            float,
        )
        self.transfer_hand = np.asarray(
            phase3_config.get("transfer_hand", phase1_config["hand_close"]),
            float,
        )
        self.state = "UNCAP_ALIGN"
        self.reason = ""
        self.state_steps = 0
        self.contaminated = False
        self.drops = 0
        self._dropped = False
        self.port = None
        self._last_target = None
        self._current_quat = None
        self._uncap_sweep_start = None
        self._cap_start_angle = 0.0
        self._uncap_qpos = None
        self._uncap_qvel = None
        self._uncap_ctrl = None
        self._vial_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "vial_body")
        self._mat_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "contam_mat")
        self._port_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "port_center")
        self._decapper_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "decapper_center")
        self._cap_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cap_geom")

    def run(self):
        while self.phase1.state not in {"HOLD", "FAILED"}:
            self.phase1.step()
        if self.phase1.state == "FAILED":
            return Phase3Result(False, "GRASP", self.phase1.reason, self.metrics())
        self.port = self.env.data.site_xpos[self._port_site].copy()
        self._last_target = self.env.ee_pos()
        self._current_quat = self.phase1.quat.copy()
        self._cap_start_angle = self._cap_angle()
        while self.state not in {"DONE", "FAILED"}:
            self.step()
        metrics = self.metrics()
        passed = self.state == "DONE" and metrics.uncapped and not metrics.contaminated and metrics.drops == 0
        return Phase3Result(passed, self.state, self.reason, metrics)

    def step(self):
        if self.state in {"UNCAP_ALIGN", "UNCAP_SWEEP", "TRANSFER", "LOWER"} and self._contaminated():
            self.contaminated = True
            self._fail("vial contacted the contamination mat")
            return
        if self.state in {"UNCAP_ALIGN", "UNCAP_SWEEP"} and self._uncontrolled_drop():
            self.drops = 1
            self._fail("vial dropped before sealing")
            return
        if self.state == "UNCAP_ALIGN":
            self._hold(self._uncap_target(), self.uncap_hand, self.phase1.quat)
            if self.state_steps >= self.config["uncap_align_ready_steps"] and self._cap_aligned():
                self._uncap_sweep_start = self._last_target.copy()
                self._transition("UNCAP_SWEEP")
            elif self.state_steps >= self.config["uncap_align_max_steps"]:
                self._fail("cap did not engage the decapper")
        elif self.state == "UNCAP_SWEEP":
            progress = min(1.0, (self.state_steps + 1) / self.config["uncap_sweep_steps"])
            self._current_quat = self.phase1.quat
            self._hold(self._uncap_sweep_target(progress), self.uncap_hand, self.phase1.quat)
            if self._uncapped() and self.state_steps >= self.config["uncap_min_sweep_steps"]:
                self._capture_uncap_snapshot()
                self._transition("TRANSFER")
            elif self.state_steps >= self.config["uncap_sweep_max_steps"]:
                self._fail("cap did not rotate past the uncap angle")
        elif self.state == "TRANSFER":
            self._hold(self._servo_target(self.config["transfer_height"]), self.transfer_hand)
            if self.state_steps >= self.config["transfer_ready_steps"] and self._vial_over_port():
                self._transition("LOWER")
            elif self.state_steps >= self.config["transfer_max_steps"]:
                self._fail("did not reach the waste port")
        elif self.state == "LOWER":
            self._hold(self._servo_target(self.config["lower_height"]), self.transfer_hand)
            if self._vial_in_port() or self.state_steps >= self.config["lower_steps"]:
                self._transition("RELEASE")
        elif self.state == "RELEASE":
            self._hold(self._last_target, self.config["release_hand"])
            if self.state_steps >= self.config["release_steps"]:
                self._transition("RETRACT")
        elif self.state == "RETRACT":
            self._hold(self._last_target + np.array([0.0, 0.0, self.config["retract_height"]]), self.config["release_hand"])
            if self._ee_retracted() and self.state_steps >= self.config["retract_steps"] // 2:
                if self._vial_in_port():
                    self.reason = "vial sealed in the waste port"
                    self.state = "DONE"
                else:
                    self._fail("vial left the port after release")
            elif self.state_steps >= self.config["retract_steps"]:
                self._fail("end-effector did not retract")
        self.state_steps += 1

    def restore_uncap_snapshot(self):
        if self._uncap_qpos is None:
            return False
        self.env.data.qpos[:] = self._uncap_qpos
        self.env.data.qvel[:] = self._uncap_qvel
        self.env.data.ctrl[:] = self._uncap_ctrl
        mujoco.mj_forward(self.env.model, self.env.data)
        return True

    def metrics(self):
        vial = self.env.sensor("vial_pos")
        port_distance = float(np.linalg.norm(vial[:2] - self.port[:2])) if self.port is not None else float("inf")
        return Phase3Metrics(
            vial_in_port=self._vial_in_port() if self.port is not None else False,
            ee_retracted=self._ee_retracted() if self.port is not None else False,
            contaminated=self.contaminated,
            uncapped=self._uncapped(),
            drops=self.drops,
            cap_angle_deg=round(self._cap_angle_deg(), 1),
            cap_delta_deg=round(self._cap_delta_deg(), 1),
            uncap_angle_deg=float(self.config["uncap_angle_deg"]),
            port_distance=port_distance,
            final_vial_pos=tuple(round(float(x), 3) for x in vial),
        )

    def _servo_target(self, height):
        bias = np.asarray(self.config.get("transfer_xy_bias", [0.0, 0.0]), float)
        desired_vial = np.array([self.port[0] + bias[0], self.port[1] + bias[1], self.port[2] + height])
        self._last_target = desired_vial + (self.env.ee_pos() - self.env.sensor("vial_pos"))
        return self._last_target

    def _hold(self, target, hand, quat=None):
        self.env.data.ctrl[self.env._hand_ctrl] = hand
        quat_des = self._current_quat if quat is None else quat
        self.impedance.apply(target, quat_des)
        self.env.step()

    def _uncap_target(self):
        center = self.env.data.site_xpos[self._decapper_site].copy()
        cap = self.env.data.geom_xpos[self._cap_geom].copy()
        self._last_target = center + (self.env.ee_pos() - cap)
        return self._last_target

    def _uncap_sweep_target(self, progress):
        target = self._uncap_sweep_start + np.asarray(self.config["uncap_sweep_offset"], float) * progress
        self._last_target = target
        return target

    def _cap_aligned(self):
        center = self.env.data.site_xpos[self._decapper_site]
        cap = self.env.data.geom_xpos[self._cap_geom]
        return float(np.linalg.norm(cap - center)) <= self.config["uncap_align_tol"]

    def _cap_angle(self):
        return float(self.env.sensor("cap_thread_pos")[0])

    def _cap_angle_deg(self):
        return float(np.degrees(self._cap_angle()))

    def _cap_delta_deg(self):
        return float(np.degrees(abs(self._cap_angle() - self._cap_start_angle)))

    def _uncapped(self):
        return self._cap_angle_deg() >= self.config["uncap_angle_deg"]

    def _capture_uncap_snapshot(self):
        if self._uncap_qpos is None:
            self._uncap_qpos = self.env.data.qpos.copy()
            self._uncap_qvel = self.env.data.qvel.copy()
            self._uncap_ctrl = self.env.data.ctrl.copy()

    def _transition(self, state):
        self.state = state
        self.state_steps = 0

    def _fail(self, reason):
        self.reason = reason
        self.state = "FAILED"

    def _vial_over_port(self):
        vial = self.env.sensor("vial_pos")
        return float(np.linalg.norm(vial[:2] - self.port[:2])) <= self.config["port_xy_tol"]

    def _vial_in_port(self):
        vial = self.env.sensor("vial_pos")
        in_xy = float(np.linalg.norm(vial[:2] - self.port[:2])) <= self.config["port_xy_tol"]
        in_z = vial[2] <= self.port[2] + self.config["sealed_z_margin"]
        return in_xy and in_z

    def _ee_retracted(self):
        return self.env.ee_pos()[2] >= self.port[2] + self.config["retract_height"] - 0.05

    def _uncontrolled_drop(self):
        if self.port is not None and self._vial_in_port():
            return False
        lift = float(self.env.sensor("vial_pos")[2] - self.phase1.start_vial_z)
        dropped = (
            lift < self.config["drop_lift"]
            and self.contacts.fingers_engaged(self.phase1_config["f_grasp_min"]) < self.phase1_config["min_fingers"]
        )
        if dropped and not self._dropped:
            self._dropped = True
        return dropped

    def _contaminated(self):
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if self._vial_geom in geoms and self._mat_geom in geoms:
                return True
        return False


@dataclass(frozen=True)
class Phase4Metrics:
    yaw_deg: float
    target_yaw_deg: float
    min_lift: float
    max_tilt_deg: float
    min_fingers: int
    force_closure_misses: int
    low_finger_misses: int
    dropped: bool
    force_closure: bool
    fingers_engaged: int
    command_quat_drift_deg: float
    actual_ee_quat_drift_deg: float


@dataclass(frozen=True)
class Phase4Result:
    passed: bool
    state: str
    reason: str
    metrics: Phase4Metrics


class Phase4FSM:
    def __init__(self, env, impedance, contacts, phase1_config, phase4_config):
        self.env = env
        self.impedance = impedance
        self.contacts = contacts
        self.phase1_config = phase1_config
        self.config = phase4_config
        self.phase1 = Phase1FSM(env, impedance, contacts, phase1_config)
        self.state = "REORIENT"
        self.reason = ""
        self.state_steps = 0
        self.start_yaw = None
        self.start_ee_quat = None
        self.fixed_quat = None
        self.fixed_target = None
        self.base_hand = np.asarray(phase1_config["hand_close"], float)
        self.reorient_hand = self._bounded_hand(
            self.base_hand + np.asarray(phase4_config["reorient_delta"], float),
        )
        self.min_lift = float("inf")
        self.max_tilt = 0.0
        self.min_fingers = 4
        self.force_closure_misses = 0
        self.low_finger_misses = 0
        self.max_actual_ee_drift = 0.0

    def run(self):
        while self.phase1.state not in {"HOLD", "FAILED"}:
            self.phase1.step()
        if self.phase1.state == "FAILED":
            return Phase4Result(False, "GRASP", self.phase1.reason, self.metrics())
        self._apply_phase4_impedance()
        self.start_yaw = self._yaw()
        self.start_ee_quat = self.env.ee_quat()
        self.fixed_quat = self.start_ee_quat.copy()
        self.fixed_target = self.phase1.lift.copy()
        while self.state not in {"DONE", "FAILED"}:
            self.step()
        metrics = self.metrics()
        passed = (
            self.state == "DONE"
            and metrics.yaw_deg >= metrics.target_yaw_deg
            and not metrics.dropped
            and metrics.max_tilt_deg <= self.config["max_tilt_deg"]
            and metrics.min_fingers >= self.config["min_fingers"]
            and metrics.force_closure_misses == 0
            and metrics.low_finger_misses == 0
        )
        return Phase4Result(passed, self.state, self.reason, metrics)

    def step(self):
        ramp_steps = self.config["ramp_steps"]
        total_steps = ramp_steps + self.config["hold_steps"]
        alpha = min(1.0, (self.state_steps + 1) / ramp_steps)
        hand = self.base_hand + alpha * (self.reorient_hand - self.base_hand)
        self.env.data.ctrl[self.env._hand_ctrl] = hand
        self.impedance.apply(self.fixed_target, self.fixed_quat)
        self.env.step()
        self._track_metrics()
        if self._dropped():
            self._fail("vial dropped during reorient")
        elif self.state_steps + 1 >= total_steps:
            metrics = self.metrics()
            if (
                metrics.yaw_deg >= metrics.target_yaw_deg
                and metrics.max_tilt_deg <= self.config["max_tilt_deg"]
                and metrics.min_fingers >= self.config["min_fingers"]
                and metrics.force_closure_misses == 0
                and metrics.low_finger_misses == 0
            ):
                self.reason = "vial reoriented in hand"
                self.state = "DONE"
            else:
                self._fail("in-hand reorient did not meet the target")
        self.state_steps += 1

    def metrics(self):
        min_lift = self.min_lift
        if not np.isfinite(min_lift):
            min_lift = self._lifted_height()
        return Phase4Metrics(
            yaw_deg=self._yaw_delta_deg(),
            target_yaw_deg=float(self.config["target_yaw_deg"]),
            min_lift=min_lift,
            max_tilt_deg=self.max_tilt,
            min_fingers=self.min_fingers,
            force_closure_misses=self.force_closure_misses,
            low_finger_misses=self.low_finger_misses,
            dropped=self._dropped(),
            force_closure=self.contacts.has_force_closure(self.phase1_config["f_grasp_min"]),
            fingers_engaged=self.contacts.fingers_engaged(self.phase1_config["f_grasp_min"]),
            command_quat_drift_deg=0.0,
            actual_ee_quat_drift_deg=self.max_actual_ee_drift,
        )

    def _track_metrics(self):
        lift = self._lifted_height()
        self.min_lift = min(self.min_lift, lift)
        self.max_tilt = max(self.max_tilt, self._tilt_deg())
        fingers = self.contacts.fingers_engaged(self.phase1_config["f_grasp_min"])
        self.min_fingers = min(self.min_fingers, fingers)
        if fingers < self.config["min_fingers"]:
            self.low_finger_misses += 1
        if not self.contacts.has_force_closure(self.phase1_config["f_grasp_min"]):
            self.force_closure_misses += 1
        self.max_actual_ee_drift = max(self.max_actual_ee_drift, self._actual_ee_drift_deg())

    def _bounded_hand(self, hand):
        low = self.env.model.actuator_ctrlrange[self.env._hand_ctrl, 0]
        high = self.env.model.actuator_ctrlrange[self.env._hand_ctrl, 1]
        return np.clip(hand, low, high)

    def _apply_phase4_impedance(self):
        if "arm_kp_rot" in self.config:
            self.impedance.kp[3:] = np.asarray(self.config["arm_kp_rot"], float)
        if "arm_kd_rot" in self.config:
            self.impedance.kd[3:] = np.asarray(self.config["arm_kd_rot"], float)

    def _lifted_height(self):
        return float(self.env.sensor("vial_pos")[2] - self.phase1.start_vial_z)

    def _dropped(self):
        return self._lifted_height() < self.phase1_config["min_lift"]

    def _yaw(self):
        qw, qx, qy, qz = self.env.sensor("vial_quat")
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def _yaw_delta_deg(self):
        if self.start_yaw is None:
            return 0.0
        delta = np.arctan2(np.sin(self._yaw() - self.start_yaw), np.cos(self._yaw() - self.start_yaw))
        return float(abs(np.degrees(delta)))

    def _tilt_deg(self):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.env.sensor("vial_quat"))
        axis_alignment = abs(rotation.reshape(3, 3)[2, 2])
        return float(np.degrees(np.arccos(np.clip(axis_alignment, -1.0, 1.0))))

    def _actual_ee_drift_deg(self):
        if self.start_ee_quat is None:
            return 0.0
        error = np.zeros(3)
        mujoco.mju_subQuat(error, self.start_ee_quat, self.env.ee_quat())
        return float(np.degrees(np.linalg.norm(error)))

    def _fail(self, reason):
        self.reason = reason
        self.state = "FAILED"
