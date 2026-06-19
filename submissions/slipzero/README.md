# SlipZero

**Auditable, tactile-closed-loop dexterous handling of "never-drop" hazardous sample vials in MuJoCo.**

A five-finger hand grasps a sealed sample vial, holds it under active perturbations, recovers from
incipient slip, and transfers it to a waste-disposal port — with every state transition driven by a
**sensor reading**, never a wall-clock timer.

## Robot platform
- **Arm:** Franka Emika Panda (7 DOF)
- **Hand:** LEAP Hand (16 DOF), dexterous five-finger
- Composed at load time from MuJoCo Menagerie assets (see `assets/ATTRIBUTION.md`).

We evaluated the FFAI starter platforms (Aegis quadruped, Futurist and FF Master humanoids) and
chose a dedicated 16-DOF dexterous hand instead: this task is fine in-hand manipulation — grasping a
24 mm vial, recovering from slip, and unscrewing a cap — which needs articulated fingers and a
friction-cone tactile signal that the FFAI locomotion platforms (wrist + fist, no fingers) do not
provide.

## Task goal
Robustly handle a hazardous sample vial without ever dropping or contaminating it: approach → grasp →
lift → hold under perturbation (detect and recover from slip) → transfer to a waste port → seal — and
make the result **reproducible and verifiable**.

## Technical approach
- **Cartesian impedance arm control** — task-space PD on a hand grasp site with gravity compensation,
  mapped to joint torques through the site Jacobian; joint-space damping tames the 7-DOF nullspace.
- **Real contact-force sensing** — fingertip↔vial contacts are read via `mj_contactForce` and
  decomposed into normal/tangential components; the **friction-cone margin** (`μ·fₙ − ‖fₜ‖`) is the
  core tactile signal.
- **Sensor-gated state machines** — every success transition fires on a sensor (end-effector pose,
  per-finger normal force, force closure, friction-cone margin, vial pose). Timeouts exist only as
  failure guards.

## Core features (current)
- **Phase 0** — composed Panda+LEAP rig with touch / wrist force-torque / vial-pose sensors.
- **Phase 1** — impedance grasp → lift → upright 5 s hold, gated on force closure.
- **Phase 2** — incipient-slip detection from the friction-cone margin; a perturbation that drops the
  vial under a fixed grip is **recovered** by sensor-triggered grip escalation (baseline drops,
  closed loop holds; recovery latency logged).
- **Phase 3** — transfer the grasped vial to a waste port and seal it, with fail-on-contamination.

## Run instructions
From this folder (`submissions/slipzero/`), on Python 3.11:

```bash
# create an isolated env and install pinned deps
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# headless self-checks (deterministic, no display needed)
python demo.py --headless      # Phase 0: rig + sensors respond to contact
python phase1.py               # Phase 1: grasp -> lift -> hold
python phase2.py               # Phase 2: slip detection + recovery (baseline vs closed loop)
python phase3.py               # Phase 3: transfer + seal in the waste port

# interactive viewer (macOS needs mjpython for the window)
mjpython demo.py
```
Each check prints a `PHASE n: PASS/FAIL` line and writes a render to `results/`.

## Highlights
- The slip loop closes on **genuine MuJoCo contact forces**, not a scripted timeline.
- Transitions are provably **sensor-gated**, and every check is **deterministic** (same seed → same
  result), which is the foundation for an auditable N-trial evaluation.

## Current limitations
- **Uncap** (rotating the threaded cap off the vial) is in progress.
- The **N-trial randomized evaluation harness** (headline success rate / drop count over seeded
  trials, with an open-loop baseline row) and the **demo video** are not yet included.
- Phase 2's slip recovery is demonstrated at a deterministic operating point; robustness across a
  full domain-randomization sweep is future work.

_Demo video: to be added (`demo.mp4` / link)._
