from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from phase1 import _controller, save_png
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase3FSM
from slipzero.sensors import ContactReader


def _load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    phase1 = dict(config["phase1"])
    phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    phase3 = dict(config["phase3"])
    return config, phase1, phase3


def run(config_path, render=True):
    config, phase1, phase3 = _load_config(config_path)
    env = SlipZeroEnv(arm_torque=True)
    fsm = Phase3FSM(env, _controller(env, config), ContactReader(env), phase1, phase3)
    result = fsm.run()
    if render:
        save_png(phase3["render_path"], env, phase3["render"])
        if fsm.restore_uncap_snapshot():
            save_png(phase3["uncap_render_path"], env, phase3["render"])
    render_paths = (phase3["uncap_render_path"], phase3["render_path"]) if render else None
    return result, render_paths


def main():
    parser = argparse.ArgumentParser(description="SlipZero Phase 3 transfer/seal check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    result, render_path = run(args.config, render=not args.no_render)
    metrics = result.metrics
    print(f"final state: {result.state} ({result.reason})")
    print(
        f"uncapped: {metrics.uncapped}  cap angle: {metrics.cap_angle_deg:.1f} deg "
        f"delta: {metrics.cap_delta_deg:.1f} deg  target: {metrics.uncap_angle_deg:.1f} deg"
    )
    print(f"vial in port: {metrics.vial_in_port}  port distance: {metrics.port_distance:.3f} m")
    print(f"ee retracted: {metrics.ee_retracted}  drops: {metrics.drops}  contaminated: {metrics.contaminated}")
    print(f"final vial pos: {metrics.final_vial_pos}")
    if render_path is not None:
        print(f"uncap render: {render_path[0]}")
        print(f"seal render: {render_path[1]}")
    print("PHASE 3:", "PASS — vial transferred and sealed in the waste port" if result.passed else "FAIL")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
