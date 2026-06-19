from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from slipzero.env import EYE_IN_HAND_CAM, WORKSPACE_CAM


@dataclass(frozen=True)
class VisionReading:
    camera: str
    vial_visible: bool
    area_fraction: float
    centroid_uv: tuple | None
    mean_depth: float
    grasp_confidence: float


class VisionSensor:
    def __init__(self, env, width, height, visible_area_min, confidence_area_ref):
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.visible_area_min = float(visible_area_min)
        self.confidence_area_ref = float(confidence_area_ref)
        self._renderer = mujoco.Renderer(env.model, self.height, self.width)
        self._vial_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "vial_body")

    def rgb(self, camera):
        self._renderer.disable_depth_rendering()
        self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self.env.data, camera=camera)
        return self._renderer.render()

    def vial_mask(self, camera):
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.env.data, camera=camera)
        objid = self._renderer.render()[:, :, 0]
        self._renderer.disable_segmentation_rendering()
        return objid == self._vial_geom

    def _depth(self, camera):
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.env.data, camera=camera)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return depth

    def read(self, camera):
        mask = self.vial_mask(camera)
        pixels = int(mask.sum())
        area_fraction = pixels / mask.size
        visible = area_fraction >= self.visible_area_min
        if pixels:
            ys, xs = np.nonzero(mask)
            centroid = (float(xs.mean()) / self.width, float(ys.mean()) / self.height)
            mean_depth = float(self._depth(camera)[mask].mean())
        else:
            centroid = None
            mean_depth = float("nan")
        confidence = float(np.clip(area_fraction / self.confidence_area_ref, 0.0, 1.0))
        return VisionReading(camera, visible, area_fraction, centroid, mean_depth, confidence)

    def read_workspace(self):
        return self.read(WORKSPACE_CAM)

    def read_eye_in_hand(self):
        return self.read(EYE_IN_HAND_CAM)
