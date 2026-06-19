from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from phase1 import _controller
from slipzero.env import EYE_IN_HAND_CAM, SlipZeroEnv, WORKSPACE_CAM
from slipzero.fsm import Phase1FSM
from slipzero.fusion import GraspMonitor
from slipzero.sensors import ContactReader
from slipzero.vision import VisionSensor


def _save_overlay(path, rgb, mask):
    image = np.ascontiguousarray(rgb).copy()
    tint = np.array([60, 220, 120], dtype=np.uint16)
    image[mask] = np.clip(0.45 * image[mask] + 0.55 * tint, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def run(config_path, save_dir):
    config = yaml.safe_load(Path(config_path).read_text())
    phase1 = dict(config["phase1"])
    phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    vcfg = config["vision"]

    env = SlipZeroEnv(arm_torque=True)
    controller = _controller(env, config)
    contacts = ContactReader(env)
    vision = VisionSensor(env, vcfg["width"], vcfg["height"],
                          vcfg["visible_area_min"], vcfg["confidence_area_ref"])
    monitor = GraspMonitor(env, contacts, vision,
                           phase1["f_grasp_min"], phase1["min_fingers"], vcfg["slip_area_drop"])

    env.reset()
    grasp = Phase1FSM(env, controller, contacts, phase1)
    while grasp.state not in {"HOLD", "FAILED"}:
        grasp.step()
    if grasp.state == "FAILED":
        print(f"VISION: FAIL (grasp did not reach hold: {grasp.reason})")
        return False

    held = monitor.update()
    eih_grasp_area = held.eye_in_hand.area_fraction
    ws_grasp_area = held.workspace.area_fraction

    hold_frames = 0
    for _ in range(20):
        env.data.ctrl[env._hand_ctrl] = grasp.close_hand
        env.step(vcfg["monitor_every_steps"])
        integrity = monitor.update()
        hold_frames += 1
        if integrity.tactile_ok and integrity.vision_ok and integrity.agreement:
            confirmed = hold_frames
        else:
            confirmed = None

    rgb_eih = vision.rgb(EYE_IN_HAND_CAM)
    mask_eih = vision.vial_mask(EYE_IN_HAND_CAM)
    rgb_ws = vision.rgb(WORKSPACE_CAM)
    mask_ws = vision.vial_mask(WORKSPACE_CAM)
    _save_overlay(Path(save_dir) / "vision_eye_in_hand.png", rgb_eih, mask_eih)
    _save_overlay(Path(save_dir) / "vision_workspace.png", rgb_ws, mask_ws)

    agreement_rate = monitor.agreement_rate()

    open_hand = np.full(env._hand_ctrl.shape, float(phase1["hand_open"]))
    vision_caught_loss = False
    for _ in range(40):
        env.data.ctrl[env._hand_ctrl] = open_hand
        env.step(vcfg["monitor_every_steps"])
        integrity = monitor.update()
        if integrity.vision_slip and not integrity.vision_ok:
            vision_caught_loss = True
            break

    print(f"grasped vial area  eye-in-hand {eih_grasp_area:.4f}  workspace {ws_grasp_area:.4f}")
    print(f"hold agreement rate (tactile vs vision): {agreement_rate:.2f}")
    print(f"vision confirmed grasp during hold: {held.vision_ok and held.tactile_ok}")
    print(f"vision-only slip events: {monitor.vision_slip_events}  caught release-loss: {vision_caught_loss}")
    print(f"overlays: {save_dir}/vision_eye_in_hand.png  {save_dir}/vision_workspace.png")

    passed = (
        held.tactile_ok
        and held.vision_ok
        and held.eye_in_hand.vial_visible
        and held.workspace.vial_visible
        and agreement_rate >= 0.8
        and vision_caught_loss
    )
    print("VISION:", "PASS" if passed else "FAIL")
    return passed


def main():
    parser = argparse.ArgumentParser(description="SlipZero vision-tactile fusion check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()
    return 0 if run(args.config, args.save_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
