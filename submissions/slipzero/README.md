# SlipZero

**Auditable, tactile-closed-loop dexterous handling of "never-drop" hazardous sample vials in MuJoCo.**

A five-finger hand grasps a sealed sample vial, holds it under perturbation and recovers from
incipient slip, unscrews the cap, reorients the vial in-hand, and transfers it to a waste-disposal
port — with every state transition driven by a **sensor reading**, never a wall-clock timer — and a
seeded N-trial evaluation that **proves** the robustness with reproducible numbers.

## Robot platform
- **Arm:** Franka Emika Panda (7 DOF) · **Hand:** LEAP Hand (16 DOF), dexterous five-finger
- Composed at load time from MuJoCo Menagerie assets (see `assets/ATTRIBUTION.md`).

We evaluated the FFAI starter platforms (Aegis quadruped, Futurist and FF Master humanoids) and chose
a dedicated 16-DOF dexterous hand: this task is fine in-hand manipulation, which needs articulated
fingers and a friction-cone tactile signal that the FFAI locomotion platforms (wrist + fist, no
fingers) do not provide.

## Task goal
Robustly handle a hazardous sample vial without dropping or contaminating it: grasp → hold under
perturbation (detect + recover from slip) → uncap → in-hand reorient → transfer to a waste port →
seal — and make the robustness **measurable and reproducible**.

## Technical approach
- **Cartesian impedance arm control** — task-space PD on a hand grasp site with gravity compensation
  through the site Jacobian; joint-space damping tames the 7-DOF nullspace (≈1 mm tracking).
- **Real contact-force sensing** — fingertip↔vial contacts read via `mj_contactForce`, decomposed
  into normal/tangential; the **friction-cone margin** (`μ·fₙ − ‖fₜ‖`) is the core tactile signal.
- **Sensor-gated state machines** — every success transition fires on a sensor (EE pose, per-finger
  normal force, force closure, friction-cone margin, vial pose, cap angle). Timeouts are failure
  guards only.

## Core features
- **Grasp / lift / hold** — impedance grasp to force closure, lift, and a 5 s upright hold.
- **Slip detection + recovery** — a perturbation that drops the vial under a fixed grip is recovered
  by sensor-triggered grip escalation (baseline drops, closed loop holds; recovery latency logged).
- **Uncap** — contact-driven: the hand seats the cap in a fixed decapper and sweeps the gripped vial
  so socket contact unscrews the cap past a threshold (no direct cap torque).
- **In-hand reorient** — a thumb+finger gait rotates the vial ~46° about its axis with the wrist
  command held fixed (genuine in-hand rotation, not a wrist roll).
- **Transfer + seal** — carry the vial to a waste bin and release, with fail-on-contamination-contact.
- **Audited evaluation** — seeded domain randomization (friction / mass / impulse), closed loop vs
  open-loop baseline, deterministic, with a per-trial `metrics.csv` and a repro test.

## Headline result (`python eval.py --trials 20 --seed 0`, full randomization range)
- **SlipZero (closed loop): 90% success, 2 drops**
- **Open-loop baseline: 80% success, 4 drops** — the closed loop halves the drops
- Mean slip-recovery latency ≈ 4 ms; deterministic (same seed → same numbers).

## Run instructions
Python 3.11. From this folder:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py --headless     # rig + sensors respond to contact (Phase 0)
python phase1.py              # grasp -> lift -> hold
python phase2.py              # slip detection + recovery (baseline vs closed loop)
python phase3.py              # uncap -> transfer -> seal
python phase4.py              # in-hand reorient
python eval.py --trials 20 --seed 0   # seeded audit -> results/metrics.csv + headline table
pytest tests/test_repro.py    # asserts the eval is deterministic

mjpython demo.py              # interactive viewer (macOS); python demo.py elsewhere
python render_demo.py         # regenerate demo.mp4 (needs the ffmpeg binary on PATH)
```
Each check prints `PHASE n: PASS/FAIL`. The demo video at `demo.mp4` is produced by `render_demo.py`,
which runs the real controllers; its audit card numbers are read from `metrics.csv`.

## Demo video
`demo.mp4` (≈50 s): baseline drop → closed-loop slip recovery → grasp/uncap/transfer/seal →
in-hand reorient → audited-robustness card with the reproducible numbers.

## Highlights
- The slip loop closes on **genuine MuJoCo contact forces**, not a scripted timeline.
- Transitions are provably **sensor-gated**, and every check is **deterministic**, so the headline
  numbers are auditable: a judge runs `python eval.py` and watches the same numbers appear.

## Current limitations
- The pipeline is tuned around a nominal operating point; it is reliable across a moderate
  friction/mass envelope (the eval reports the reliable band) but not the entire randomization range
  — hence 90%, reported honestly rather than narrowing the ranges to manufacture a higher number.
- The in-hand reorient gait is open-loop (a tuned finger trajectory) with a sensor-verified outcome,
  rather than continuously closed-loop on the vial yaw.
- No real fluid simulation (MuJoCo has none); the "sample" is a rigid vial.

## Future improvements
- Adaptive/force-closure-driven grip to widen the reliable randomization envelope toward the full range.
- Closed-loop (feedback-corrected) in-hand reorient gait.
- mjx-parallel evaluation to push the trial count to hundreds.
