from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

UUID_FORMAT = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
ABSOLUTE_PATH = re.compile(r"(/Users/|/home/[A-Za-z]|[A-Za-z]:\\\\)")

REQUIRED_FILES = (
    "registration.json", "README.md", "LICENSE", "requirements.txt", "run.sh",
    "demo.py", "phase1.py", "phase2.py", "phase3.py", "phase4.py", "vision_check.py",
    "eval.py", "render_demo.py", "config/default.yaml", "tests/test_repro.py",
    "slipzero/env.py", "slipzero/control.py", "slipzero/sensors.py",
    "slipzero/vision.py", "slipzero/fusion.py", "slipzero/fsm.py",
    "assets/slipzero_bench.xml", "demo.mp4",
)
CONFIG_SECTIONS = ("thresholds", "controller", "randomization", "vision",
                   "phase1", "phase2", "phase3", "phase4")
PACKAGE_MODULES = ("slipzero.env", "slipzero.control", "slipzero.sensors",
                   "slipzero.vision", "slipzero.fusion", "slipzero.fsm")
SOURCE_GLOBS = ("*.py", "slipzero/*.py", "tests/*.py")


def registration_uuid():
    data = json.loads((ROOT / "registration.json").read_text())
    uuid = str(data.get("uuid", "")).strip()
    ok = bool(UUID_FORMAT.match(uuid))
    return ok, f"{uuid or '<missing>'}"


def required_files():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    return not missing, "all present" if not missing else f"missing {missing}"


def pinned_dependencies():
    unpinned = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "==" not in stripped:
            unpinned.append(stripped)
    return not unpinned, "every dependency pinned" if not unpinned else f"unpinned {unpinned}"


def config_complete():
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    absent = [section for section in CONFIG_SECTIONS if section not in config]
    return not absent, "all sections present" if not absent else f"missing {absent}"


def package_imports():
    for module in PACKAGE_MODULES:
        importlib.import_module(module)
    return True, f"{len(PACKAGE_MODULES)} modules import"


def vision_cameras():
    from slipzero.env import EYE_IN_HAND_CAM, SlipZeroEnv, WORKSPACE_CAM
    env = SlipZeroEnv()
    names = {env.model.camera(i).name for i in range(env.model.ncam)}
    present = {EYE_IN_HAND_CAM, WORKSPACE_CAM} <= names
    return present, f"cameras {sorted(names)}"


def portable_paths():
    offenders = []
    for pattern in SOURCE_GLOBS:
        for source in ROOT.glob(pattern):
            if source.name == Path(__file__).name:
                continue
            if ABSOLUTE_PATH.search(source.read_text()):
                offenders.append(source.relative_to(ROOT).as_posix())
    return not offenders, "no hardcoded absolute paths" if not offenders else f"absolute paths in {offenders}"


def deterministic_eval(trials):
    audit = importlib.import_module("eval")
    config_path = str(ROOT / "config" / "default.yaml")
    first = audit.run(trials, 0, config_path)
    second = audit.run(trials, 0, config_path)
    return first == second, f"{trials} trials reproduce identically at seed 0"


def audit_integrity():
    candidates = [ROOT / "evaluation_report.json", ROOT / "results" / "evaluation_report.json"]
    report_path = next((path for path in candidates if path.exists()), None)
    if report_path is None:
        return True, "skipped (run eval.py to produce evaluation_report.json)"
    report = json.loads(report_path.read_text())
    rows = report["per_trial"]
    total = len(rows)
    drops = sum(int(row["dropped"]) for row in rows)
    success_rate = 100.0 * (total - drops) / total if total else 0.0
    ok = drops == report["drops"] and abs(success_rate - report["success_rate"]) < 1e-6
    return ok, f"report summary matches its {total} per-trial rows ({report['success_rate']:.0f}% / {report['drops']} drops)"


def main():
    parser = argparse.ArgumentParser(description="Validate the SlipZero submission against its own invariants")
    parser.add_argument("--trials", type=int, default=3, help="trials for the determinism check")
    parser.add_argument("--full", action="store_true", help="run the determinism check over the full 50-trial audit")
    args = parser.parse_args()
    trials = 50 if args.full else args.trials

    checks = [
        ("registration UUID well-formed", registration_uuid),
        ("required files present", required_files),
        ("dependencies pinned", pinned_dependencies),
        ("config schema complete", config_complete),
        ("package imports clean", package_imports),
        ("dual cameras in scene", vision_cameras),
        ("no hardcoded absolute paths", portable_paths),
        (f"deterministic eval ({trials} trials, seed 0)", lambda: deterministic_eval(trials)),
        ("audit report self-consistent", audit_integrity),
    ]

    failures = 0
    for name, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")

    print("VALIDATE:", "PASS" if not failures else f"FAIL ({failures} check(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
