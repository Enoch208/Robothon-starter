from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from slipzero.env import SlipZeroEnv, TOUCH_SENSORS
from slipzero.vision import VisionSensor

PROBE_TARGET = 1.3
SETTLE_STEPS = 120
PRESS_STEPS = 500
HOLD_STEPS = 200
REPORT_EVERY = 200
CONTACT_NEWTONS = 0.05


def _readout(env):
    touch = env.touch_forces()
    force = np.linalg.norm(env.sensor("wrist_force"))
    torque = np.linalg.norm(env.sensor("wrist_torque"))
    vial = env.sensor("vial_pos")
    return (
        f"touch[N] if={touch[0]:5.2f} mf={touch[1]:5.2f} rf={touch[2]:5.2f} th={touch[3]:5.2f}  "
        f"|wrist_F|={force:5.2f}N |wrist_T|={torque:5.2f}Nm  vial=({vial[0]:.2f},{vial[1]:.2f},{vial[2]:.2f})"
    )


def _press_target(step):
    if step < SETTLE_STEPS:
        return 0.0
    return PROBE_TARGET * min(1.0, (step - SETTLE_STEPS) / PRESS_STEPS)


def _drop_point(env):
    point = env.grasp_center()
    point[2] = 0.47
    return point


def contact_probe(env, on_step=None):
    env.reset()
    env.place_vial(_drop_point(env))
    total = SETTLE_STEPS + PRESS_STEPS + HOLD_STEPS
    peak_touch = np.zeros(len(TOUCH_SENSORS))
    for step in range(total):
        env.set_hand_target(_press_target(step))
        env.step()
        peak_touch = np.maximum(peak_touch, env.touch_forces())
        if on_step is not None:
            on_step(step)
    return peak_touch


def run_headless():
    env = SlipZeroEnv()
    print(f"model loaded: nq={env.model.nq} nu={env.model.nu} nsensor={env.model.nsensor}")

    def report(step):
        if step % REPORT_EVERY == 0:
            print(f"  step {step:4d}  {_readout(env)}")

    peak_touch = contact_probe(env, on_step=report)
    fingers_in_contact = int(np.sum(peak_touch > CONTACT_NEWTONS))
    wrist_force = np.linalg.norm(env.sensor("wrist_force"))

    vcfg = yaml.safe_load(Path("config/default.yaml").read_text())["vision"]
    vision = VisionSensor(env, vcfg["width"], vcfg["height"],
                          vcfg["visible_area_min"], vcfg["confidence_area_ref"])
    workspace = vision.read_workspace()
    eye_in_hand = vision.read_eye_in_hand()
    cameras_seeing = int(workspace.vial_visible) + int(eye_in_hand.vial_visible)

    print(f"peak touch per finger [N]: {np.array2string(peak_touch, precision=2)}")
    print(f"fingers that registered contact: {fingers_in_contact}/{len(TOUCH_SENSORS)}  |  wrist |F|={wrist_force:.2f} N")
    print(f"vision: workspace area={workspace.area_fraction:.3f} ({'visible' if workspace.vial_visible else 'absent'})  "
          f"eye-in-hand area={eye_in_hand.area_fraction:.3f} ({'visible' if eye_in_hand.vial_visible else 'absent'})")
    ok = fingers_in_contact >= 1 and wrist_force > 0.0 and cameras_seeing >= 1
    print("PHASE 0:", "PASS — touch, wrist force/torque, and dual-camera vision all register the vial" if ok else "FAIL")
    return 0 if ok else 1


def run_viewer():
    import mujoco.viewer

    env = SlipZeroEnv()
    print(f"model loaded: nq={env.model.nq} nu={env.model.nu} nsensor={env.model.nsensor}")
    env.reset()
    env.place_vial(_drop_point(env))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            env.set_hand_target(_press_target(step))
            env.step()
            if step % REPORT_EVERY == 0:
                print(f"  step {step:4d}  {_readout(env)}")
            viewer.sync()
            step += 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="SlipZero Phase 0 instrumented demo")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window and self-check the sensors (no display needed)")
    args = parser.parse_args()

    if args.headless:
        return run_headless()
    try:
        return run_viewer()
    except RuntimeError as exc:
        print(f"interactive viewer unavailable ({exc}).", file=sys.stderr)
        print("on macOS run `mjpython demo.py`, or use `python demo.py --headless`.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
