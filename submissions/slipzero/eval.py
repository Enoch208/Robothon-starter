from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

from phase1 import _controller
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase2FSM
from slipzero.sensors import ContactReader

_FIELDS = [
    "trial", "seed", "vial_friction", "vial_mass", "impulse_n",
    "success", "dropped", "slip_events", "recovery_ms", "peak_grip_n", "baseline_dropped",
]


def _run_scenario(env, controller, contacts, phase1, phase2, recovery_enabled):
    env.reset()
    fsm = Phase2FSM(env, controller, contacts, phase1, phase2, recovery_enabled=recovery_enabled)
    return fsm.run()


def run(trials, seed, config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    phase1 = dict(config["phase1"])
    phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    base_phase2 = dict(config["phase2"])
    base_phase2["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    rand = config["randomization"]
    friction_range = tuple(rand["friction"])
    mass_range = tuple(rand["mass"])
    impulse_range = tuple(rand["impulse_newtons"])

    env = SlipZeroEnv(arm_torque=True)
    controller = _controller(env, config)
    contacts = ContactReader(env)

    rows = []
    for i in range(trials):
        rng = np.random.default_rng(seed * 1_000_003 + i)
        friction = float(rng.uniform(*friction_range))
        mass = float(rng.uniform(*mass_range))
        impulse = float(rng.uniform(*impulse_range))
        phase2 = dict(base_phase2)
        phase2["impulse_force"] = [0.0, 0.0, -impulse]

        env.set_vial_params(friction, mass)
        closed = _run_scenario(env, controller, contacts, phase1, phase2, True)

        env.set_vial_params(friction, mass)
        baseline = _run_scenario(env, controller, contacts, phase1, phase2, False)

        rows.append({
            "trial": i,
            "seed": seed,
            "vial_friction": round(friction, 4),
            "vial_mass": round(mass, 4),
            "impulse_n": round(impulse, 3),
            "success": int(not closed.metrics.dropped),
            "dropped": int(closed.metrics.dropped),
            "slip_events": closed.metrics.slip_events,
            "recovery_ms": "" if closed.metrics.recovery_latency_ms is None else round(closed.metrics.recovery_latency_ms, 1),
            "peak_grip_n": round(closed.metrics.peak_grip_force, 2),
            "baseline_dropped": int(baseline.metrics.dropped),
        })
    return rows


def summarize(rows):
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    drops = sum(r["dropped"] for r in rows)
    baseline_drops = sum(r["baseline_dropped"] for r in rows)
    latencies = [r["recovery_ms"] for r in rows if r["recovery_ms"] != ""]
    reliable = [r for r in rows if r["success"]]
    return {
        "trials": n,
        "success_rate": 100.0 * successes / n if n else 0.0,
        "drops": drops,
        "baseline_success_rate": 100.0 * (n - baseline_drops) / n if n else 0.0,
        "baseline_drops": baseline_drops,
        "mean_recovery_ms": statistics.mean(latencies) if latencies else float("nan"),
        "median_recovery_ms": statistics.median(latencies) if latencies else float("nan"),
        "reliable_friction": (min(r["vial_friction"] for r in reliable), max(r["vial_friction"] for r in reliable)) if reliable else None,
        "reliable_mass": (min(r["vial_mass"] for r in reliable), max(r["vial_mass"] for r in reliable)) if reliable else None,
    }


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="SlipZero seeded slip-recovery evaluation")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--csv", default="results/metrics.csv")
    parser.add_argument("--report", default="results/evaluation_report.json")
    args = parser.parse_args()

    rows = run(args.trials, args.seed, args.config)
    write_csv(rows, args.csv)
    s = summarize(rows)
    Path(args.report).write_text(json.dumps(
        {"trials": args.trials, "seed": args.seed, **s, "per_trial": rows}, indent=2, default=float))
    print(f"trials={s['trials']} seed={args.seed}  (vial friction/mass/impulse randomized)")
    print(f"SlipZero (closed loop):  success {s['success_rate']:.0f}%  drops {s['drops']}")
    print(f"Baseline (no recovery):  success {s['baseline_success_rate']:.0f}%  drops {s['baseline_drops']}")
    if s["mean_recovery_ms"] == s["mean_recovery_ms"]:
        print(f"recovery latency: mean {s['mean_recovery_ms']:.0f} ms  median {s['median_recovery_ms']:.0f} ms")
    if s["reliable_friction"]:
        print(f"reliable envelope: friction [{s['reliable_friction'][0]:.2f}, {s['reliable_friction'][1]:.2f}]  "
              f"mass [{s['reliable_mass'][0]:.3f}, {s['reliable_mass'][1]:.3f}] kg")
    print(f"per-trial rows: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
