from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from phase1 import _controller, save_png
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase4FSM
from slipzero.sensors import ContactReader


def _load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    phase1 = dict(config["phase1"])
    phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    phase4 = dict(config["phase4"])
    return config, phase1, phase4


def run(config_path, render=True):
    config, phase1, phase4 = _load_config(config_path)
    env = SlipZeroEnv(arm_torque=True)
    fsm = Phase4FSM(env, _controller(env, config), ContactReader(env), phase1, phase4)
    result = fsm.run()
    if render:
        save_png(phase4["render_path"], env, phase4["render"])
    return result, phase4["render_path"] if render else None


def main():
    parser = argparse.ArgumentParser(description="SlipZero Phase 4 in-hand reorientation check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    result, render_path = run(args.config, render=not args.no_render)
    metrics = result.metrics
    print(f"final state: {result.state} ({result.reason})")
    print(f"yaw: {metrics.yaw_deg:.1f} deg  target: {metrics.target_yaw_deg:.1f} deg")
    print(f"minimum lift: {metrics.min_lift:.3f} m  max tilt: {metrics.max_tilt_deg:.1f} deg")
    print(
        f"fingers engaged: {metrics.fingers_engaged}  minimum fingers: {metrics.min_fingers} "
        f"force closure: {metrics.force_closure}"
    )
    print(
        f"force-closure misses: {metrics.force_closure_misses}  "
        f"low-finger misses: {metrics.low_finger_misses}  dropped: {metrics.dropped}"
    )
    print(
        f"command quat drift: {metrics.command_quat_drift_deg:.2f} deg  "
        f"actual ee quat drift: {metrics.actual_ee_quat_drift_deg:.2f} deg"
    )
    if render_path is not None:
        print(f"render: {render_path}")
    print("PHASE 4:", "PASS" if result.passed else "FAIL")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
