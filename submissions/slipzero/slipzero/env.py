from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_BENCH_XML = _ASSETS / "slipzero_bench.xml"
_PANDA_XML = _ASSETS / "franka_emika_panda" / "panda_nohand.xml"
_LEAP_XML = _ASSETS / "leap_hand" / "right_hand.xml"

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
ARM_ACTUATORS = ("actuator1", "actuator2", "actuator3", "actuator4",
                 "actuator5", "actuator6", "actuator7")
ARM_HOME = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
ARM_READY = (-0.01919, 0.17535, 0.07569, -0.69975, -0.03922, 0.87378, -0.75613)

HAND_ACTUATORS = (
    "leap_if_mcp_act", "leap_if_rot_act", "leap_if_pip_act", "leap_if_dip_act",
    "leap_mf_mcp_act", "leap_mf_rot_act", "leap_mf_pip_act", "leap_mf_dip_act",
    "leap_rf_mcp_act", "leap_rf_rot_act", "leap_rf_pip_act", "leap_rf_dip_act",
    "leap_th_cmc_act", "leap_th_axl_act", "leap_th_mcp_act", "leap_th_ipl_act",
)
HAND_OPEN = 0.0

_FINGERTIP_TOUCH = (
    ("touch_if", "leap_if_ds", (0.0, -0.024, 0.015), 0.016),
    ("touch_mf", "leap_mf_ds", (0.0, -0.024, 0.015), 0.016),
    ("touch_rf", "leap_rf_ds", (0.0, -0.024, 0.015), 0.016),
    ("touch_th", "leap_th_ds", (0.0, -0.020, -0.013), 0.018),
)
TOUCH_SENSORS = tuple(name for name, *_ in _FINGERTIP_TOUCH)
WRIST_SITE = "attachment_site"
VIAL_JOINT = "vial_free"
VIAL_BODY = "vial"


GRASP_SITE = "grasp_site"
_GRASP_SITE_LOCAL = (0.012, 0.030, -0.020)

WORKSPACE_CAM = "workspace_cam"
EYE_IN_HAND_CAM = "eye_in_hand"
_WORKSPACE_CAM_POS = (0.34, -0.46, 0.82)
_WORKSPACE_CAM_FOVY = 38.0
_EYE_IN_HAND_POS = (0.0, 0.065, 0.0)
_EYE_IN_HAND_FOVY = 40.0


def _build_spec(arm_torque: bool) -> mujoco.MjSpec:
    scene = mujoco.MjSpec.from_file(str(_BENCH_XML))
    arm = mujoco.MjSpec.from_file(str(_PANDA_XML))
    hand = mujoco.MjSpec.from_file(str(_LEAP_XML))

    arm.site("attachment_site").attach_body(hand.worldbody, "leap_", "")
    scene.site("robot_mount").attach_body(arm.body("link0"), "", "")

    scene.body("leap_palm").add_site(
        name=GRASP_SITE, pos=list(_GRASP_SITE_LOCAL), size=[0.012, 0.0, 0.0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE, group=4, rgba=[1.0, 0.5, 0.1, 0.6],
    )

    if arm_torque:
        for name in ARM_ACTUATORS:
            actuator = scene.actuator(name)
            actuator.gainprm = [0.0] * len(actuator.gainprm)
            actuator.biasprm = [0.0] * len(actuator.biasprm)

    for name, body, pos, radius in _FINGERTIP_TOUCH:
        scene.body(body).add_site(
            name=f"{name}_site", pos=list(pos), size=[radius, 0.0, 0.0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE, group=4, rgba=[0.1, 0.9, 0.3, 0.4],
        )
        touch = scene.add_sensor()
        touch.name = name
        touch.type = mujoco.mjtSensor.mjSENS_TOUCH
        touch.objtype = mujoco.mjtObj.mjOBJ_SITE
        touch.objname = f"{name}_site"

    for name, kind in (("wrist_force", mujoco.mjtSensor.mjSENS_FORCE),
                       ("wrist_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        ft = scene.add_sensor()
        ft.name = name
        ft.type = kind
        ft.objtype = mujoco.mjtObj.mjOBJ_SITE
        ft.objname = WRIST_SITE

    workspace_cam = scene.worldbody.add_camera()
    workspace_cam.name = WORKSPACE_CAM
    workspace_cam.pos = list(_WORKSPACE_CAM_POS)
    workspace_cam.fovy = _WORKSPACE_CAM_FOVY
    workspace_cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    workspace_cam.targetbody = VIAL_BODY

    eye_in_hand = scene.body("leap_palm").add_camera()
    eye_in_hand.name = EYE_IN_HAND_CAM
    eye_in_hand.pos = list(_EYE_IN_HAND_POS)
    eye_in_hand.fovy = _EYE_IN_HAND_FOVY
    eye_in_hand.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    eye_in_hand.targetbody = VIAL_BODY

    return scene


def _name2id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class SlipZeroEnv:
    def __init__(self, arm_torque=False):
        self.arm_torque = arm_torque
        self.model = _build_spec(arm_torque).compile()
        self.data = mujoco.MjData(self.model)

        self._sensor_slice = {}
        for i in range(self.model.nsensor):
            adr = self.model.sensor_adr[i]
            self._sensor_slice[self.model.sensor(i).name] = slice(adr, adr + self.model.sensor_dim[i])

        self._arm_qadr = np.array([self.model.jnt_qposadr[_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                   for j in ARM_JOINTS])
        self._arm_ctrl = np.array([_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATORS])
        self._hand_ctrl = np.array([_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in HAND_ACTUATORS])
        self._vial_qadr = self.model.jnt_qposadr[_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, VIAL_JOINT)]
        self._vial_body = _name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, VIAL_BODY)

        self.arm_dofs = np.array([self.model.jnt_dofadr[_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in ARM_JOINTS])
        self.grasp_site_id = _name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE)

        self.reset()

    def ee_pos(self):
        return self.data.site_xpos[self.grasp_site_id].copy()

    def ee_quat(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.grasp_site_id])
        return quat

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        arm_qpos = ARM_READY if self.arm_torque else ARM_HOME
        self.data.qpos[self._arm_qadr] = arm_qpos
        self.data.ctrl[self._arm_ctrl] = arm_qpos
        self.data.ctrl[self._hand_ctrl] = HAND_OPEN
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def sensor(self, name):
        return self.data.sensordata[self._sensor_slice[name]].copy()

    def touch_forces(self):
        return np.array([self.sensor(name)[0] for name in TOUCH_SENSORS])

    def set_hand_target(self, value):
        self.data.ctrl[self._hand_ctrl] = value

    def grasp_center(self):
        tips = {name: self.data.site(f"{name}_site").xpos.copy() for name, *_ in _FINGERTIP_TOUCH}
        thumb = tips["touch_th"]
        fingers = np.mean([xpos for name, xpos in tips.items() if name != "touch_th"], axis=0)
        return 0.5 * (thumb + fingers)

    def place_vial(self, pos):
        adr = self._vial_qadr
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply_vial_force(self, force):
        self.data.xfrc_applied[self._vial_body, :3] = force

    def clear_vial_force(self):
        self.data.xfrc_applied[self._vial_body, :] = 0.0

    def set_vial_params(self, friction, mass):
        vial_geom = _name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "vial_body")
        self.model.geom_friction[vial_geom, 0] = friction
        self.model.body_mass[self._vial_body] = mass

    def randomize(self, rng, friction_range=(0.6, 1.2), mass_range=(0.03, 0.08)):
        params = {
            "vial_friction": float(rng.uniform(*friction_range)),
            "vial_mass": float(rng.uniform(*mass_range)),
        }
        self.set_vial_params(params["vial_friction"], params["vial_mass"])
        return params
