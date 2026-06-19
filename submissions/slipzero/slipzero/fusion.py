from __future__ import annotations

from dataclasses import dataclass

from slipzero.vision import VisionReading


@dataclass(frozen=True)
class GraspIntegrity:
    tactile_ok: bool
    vision_ok: bool
    agreement: bool
    vision_slip: bool
    margin: float
    workspace: VisionReading
    eye_in_hand: VisionReading


class GraspMonitor:
    def __init__(self, env, contacts, vision, f_grasp_min, min_fingers, slip_area_drop):
        self.env = env
        self.contacts = contacts
        self.vision = vision
        self.f_grasp_min = float(f_grasp_min)
        self.min_fingers = int(min_fingers)
        self.slip_area_drop = float(slip_area_drop)
        self.grasp_area_ref = None
        self.samples = 0
        self.agreement_samples = 0
        self.vision_slip_events = 0
        self._slipping = False

    def reset(self):
        self.grasp_area_ref = None
        self.samples = 0
        self.agreement_samples = 0
        self.vision_slip_events = 0
        self._slipping = False

    def update(self):
        workspace = self.vision.read_workspace()
        eye_in_hand = self.vision.read_eye_in_hand()
        tactile_ok = (
            self.contacts.fingers_engaged(self.f_grasp_min) >= self.min_fingers
            and self.contacts.has_force_closure(self.f_grasp_min)
        )
        vision_ok = eye_in_hand.vial_visible
        if tactile_ok and self.grasp_area_ref is None:
            self.grasp_area_ref = eye_in_hand.area_fraction
        vision_slip = (
            self.grasp_area_ref is not None
            and eye_in_hand.area_fraction < self.grasp_area_ref * (1.0 - self.slip_area_drop)
        )
        agreement = tactile_ok == vision_ok
        self.samples += 1
        self.agreement_samples += int(agreement)
        if vision_slip and not self._slipping:
            self.vision_slip_events += 1
        self._slipping = vision_slip
        return GraspIntegrity(
            tactile_ok=tactile_ok,
            vision_ok=vision_ok,
            agreement=agreement,
            vision_slip=vision_slip,
            margin=self.contacts.min_friction_margin(),
            workspace=workspace,
            eye_in_hand=eye_in_hand,
        )

    def agreement_rate(self):
        return self.agreement_samples / self.samples if self.samples else float("nan")
