from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from phase1 import _controller, save_png
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase2FSM
from slipzero.sensors import ContactReader


def _load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    phase1 = dict(config["phase1"])
    phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    phase2 = dict(config["phase2"])
    phase2["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    return config, phase1, phase2


def _run_trial(config, phase1, phase2, recovery_enabled):
    env = SlipZeroEnv(arm_torque=True)
    contacts = ContactReader(env)
    fsm = Phase2FSM(env, _controller(env, config), contacts, phase1, phase2, recovery_enabled)
    result = fsm.run()
    return result, fsm


def _latency_text(value):
    return "none" if value is None else f"{value:.0f} ms"


def _trial_line(label, metrics):
    return (
        f"{label}: slip events={metrics.slip_events} "
        f"latency={_latency_text(metrics.recovery_latency_ms)} "
        f"min margin={metrics.min_margin:.3f} final margin={metrics.final_margin:.3f} "
        f"peak grip={metrics.peak_grip_force:.2f} N "
        f"final lift={metrics.final_lift:.3f} m final tilt={metrics.final_tilt_deg:.1f} deg "
        f"peak tilt={metrics.max_tilt_deg:.1f} deg "
        f"fingers={metrics.fingers_engaged} force closure={metrics.force_closure} "
        f"max displacement={metrics.max_displacement:.3f} m dropped={metrics.dropped}"
    )


def run(config_path, render=True):
    config, phase1, phase2 = _load_config(config_path)
    baseline, baseline_fsm = _run_trial(config, phase1, phase2, False)
    recovery, recovery_fsm = _run_trial(config, phase1, phase2, True)
    if render:
        save_png(phase2["baseline_render_path"], baseline_fsm.env, phase2["render"])
        save_png(phase2["render_path"], recovery_fsm.env, phase2["render"])
    displacement_delta = baseline.metrics.max_displacement - recovery.metrics.max_displacement
    saved = baseline.metrics.dropped and recovery.passed and not recovery.metrics.dropped
    displacement_ok = recovery.metrics.max_displacement <= baseline.metrics.max_displacement
    passed = saved and displacement_ok
    render_paths = (phase2["baseline_render_path"], phase2["render_path"]) if render else None
    return baseline, recovery, displacement_delta, saved, displacement_ok, passed, render_paths


def main():
    parser = argparse.ArgumentParser(description="SlipZero Phase 2 slip recovery check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    baseline, recovery, displacement_delta, saved, displacement_ok, passed, render_paths = run(
        args.config,
        render=not args.no_render,
    )
    print(_trial_line("baseline recovery off", baseline.metrics))
    print(_trial_line("closed-loop recovery", recovery.metrics))
    print(
        f"margin trace: start={recovery.metrics.start_margin:.3f} "
        f"min={recovery.metrics.min_margin:.3f} final={recovery.metrics.final_margin:.3f}"
    )
    print(
        f"save gate: baseline_dropped={baseline.metrics.dropped} "
        f"recovery_dropped={recovery.metrics.dropped} saved={saved} "
        f"displacement_delta={displacement_delta:.3f} m displacement_ok={displacement_ok}"
    )
    if render_paths is not None:
        print(f"baseline render: {render_paths[0]}")
        print(f"recovery render: {render_paths[1]}")
    print("PHASE 2:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
