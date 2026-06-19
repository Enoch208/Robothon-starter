from __future__ import annotations

import mujoco
import numpy as np


class ArmImpedance:
    def __init__(self, env, kp_pos, kp_rot, kd_pos, kd_rot, ki_pos=0.0, integral_limit=8.0,
                 joint_damping=0.0):
        self.env = env
        self.kp = np.concatenate([np.asarray(kp_pos, float), np.asarray(kp_rot, float)])
        self.kd = np.concatenate([np.asarray(kd_pos, float), np.asarray(kd_rot, float)])
        self.ki_pos = float(ki_pos)
        self.integral_limit = float(integral_limit)
        self.joint_damping = float(joint_damping)
        self._integral = np.zeros(3)
        self._dt = env.model.opt.timestep
        self._jacp = np.zeros((3, env.model.nv))
        self._jacr = np.zeros((3, env.model.nv))
        self._site = env.grasp_site_id
        self._dofs = env.arm_dofs

    def reset_integral(self):
        self._integral[:] = 0.0

    def apply(self, pos_des, quat_des):
        model, data = self.env.model, self.env.data
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self._site)

        pos = data.site_xpos[self._site]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self._site])

        pos_err = np.asarray(pos_des, float) - pos
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, np.asarray(quat_des, float), quat)

        self._integral = np.clip(self._integral + pos_err * self._dt,
                                 -self.integral_limit, self.integral_limit)

        vel = self._jacp @ data.qvel
        omega = self._jacr @ data.qvel

        wrench = np.concatenate([
            self.kp[:3] * pos_err + self.ki_pos * self._integral - self.kd[:3] * vel,
            self.kp[3:] * ori_err - self.kd[3:] * omega,
        ])
        jac = np.vstack([self._jacp, self._jacr])
        tau = jac.T @ wrench

        data.qfrc_applied[self._dofs] = (
            tau[self._dofs] + data.qfrc_bias[self._dofs]
            - self.joint_damping * data.qvel[self._dofs]
        )
