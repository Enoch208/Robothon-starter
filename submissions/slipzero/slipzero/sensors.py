from __future__ import annotations

import mujoco
import numpy as np

from slipzero.env import VIAL_BODY

FINGERS = ("if", "mf", "rf", "th")
_FINGER_BODIES = {
    "if": "leap_if_ds",
    "mf": "leap_mf_ds",
    "rf": "leap_rf_ds",
    "th": "leap_th_ds",
}


class FingerContact:
    __slots__ = ("finger", "normal_force", "tangential_force", "friction_margin", "world_normal")

    def __init__(self, finger, normal_force, tangential_force, friction_margin, world_normal):
        self.finger = finger
        self.normal_force = normal_force
        self.tangential_force = tangential_force
        self.friction_margin = friction_margin
        self.world_normal = world_normal


class ContactReader:
    def __init__(self, env):
        self.env = env
        model = env.model
        self._vial_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, VIAL_BODY)
        self._finger_of_body = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body): finger
            for finger, body in _FINGER_BODIES.items()
        }
        self._wrench = np.zeros(6)

    def fingertip_contacts(self):
        model, data = self.env.model, self.env.data
        contacts = []
        for i in range(data.ncon):
            contact = data.contact[i]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            if body1 == self._vial_body:
                finger = self._finger_of_body.get(body2)
            elif body2 == self._vial_body:
                finger = self._finger_of_body.get(body1)
            else:
                finger = None
            if finger is None:
                continue
            mujoco.mj_contactForce(model, data, i, self._wrench)
            normal_force = max(float(self._wrench[0]), 0.0)
            tangential_force = float(np.hypot(self._wrench[1], self._wrench[2]))
            mu = float(contact.friction[0])
            contacts.append(FingerContact(
                finger=finger,
                normal_force=normal_force,
                tangential_force=tangential_force,
                friction_margin=mu * normal_force - tangential_force,
                world_normal=np.array(contact.frame[:3]),
            ))
        return contacts

    def finger_normal_forces(self):
        totals = {finger: 0.0 for finger in FINGERS}
        for contact in self.fingertip_contacts():
            totals[contact.finger] += contact.normal_force
        return np.array([totals[finger] for finger in FINGERS])

    def fingers_engaged(self, f_grasp_min):
        return int(np.sum(self.finger_normal_forces() >= f_grasp_min))

    def min_friction_margin(self):
        margins = [contact.friction_margin for contact in self.fingertip_contacts()]
        return min(margins) if margins else float("inf")

    def has_force_closure(self, f_grasp_min):
        engaged = [c for c in self.fingertip_contacts() if c.normal_force >= f_grasp_min]
        if len({c.finger for c in engaged}) < 3:
            return False
        units = [c.world_normal / (np.linalg.norm(c.world_normal) + 1e-9) for c in engaged]
        return float(np.linalg.norm(np.mean(units, axis=0))) < 0.9


class SlipDetector:
    def __init__(self, contacts, margin_threshold, debounce_steps=5, filter_alpha=0.25):
        self.contacts = contacts
        self.margin_threshold = float(margin_threshold)
        self.debounce_steps = int(debounce_steps)
        self.filter_alpha = float(filter_alpha)
        self.filtered_margin = None
        self._below_steps = 0

    def reset(self):
        self.filtered_margin = None
        self._below_steps = 0

    def update(self):
        raw_margin = self.contacts.min_friction_margin()
        if not np.isfinite(raw_margin):
            raw_margin = -1.0
        if self.filtered_margin is None:
            self.filtered_margin = raw_margin
        else:
            self.filtered_margin += self.filter_alpha * (raw_margin - self.filtered_margin)
        if self.filtered_margin < self.margin_threshold:
            self._below_steps += 1
        else:
            self._below_steps = 0
        return self.filtered_margin

    @property
    def slip_imminent(self):
        return self._below_steps >= self.debounce_steps
