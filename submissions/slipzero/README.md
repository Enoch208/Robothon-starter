<div align="center">

# SlipZero

**Auditable, tactile-closed-loop dexterous handling of "never-drop" hazardous sample vials in MuJoCo.**

Franka Emika Panda (7 DOF) · LEAP Hand (16 DOF) · MuJoCo 3.x

A five-finger hand grasps a sealed sample vial, **recovers from incipient slip** under perturbation,
**unscrews the cap**, **reorients the vial in-hand**, and **transfers it to a waste port** — every state
transition driven by a **sensor reading, never a wall-clock timer** — and a seeded N-trial evaluation
**proves** the robustness with numbers any judge can reproduce in one command.

<br/>

<img src="hero.png" alt="LEAP hand grasping a sample vial on the SlipZero bench" width="680"/>

</div>

---

## 🎥 Demo

https://github.com/user-attachments/assets/a790a97a-027d-4470-8eb9-7aaf580d58a7

*(≈40 s, also included as [`demo.mp4`](demo.mp4).)* The video is **produced by running the submitted
code** (`python render_demo.py`): cold-open baseline drop → tactile slip recovery (margin trace →
"RECOVERED 2 ms") → grasp/uncap/transfer/seal → in-hand reorientation → an audited-robustness card
whose numbers are read straight from `metrics.csv`, so the on-screen figures match `eval.py` exactly.

---

## 📊 Headline results

`python eval.py --trials 50 --seed 0` — full domain-randomization range (friction, mass, impulse), deterministic:

| Metric | SlipZero (closed loop) | Open-loop baseline |
|---|---|---|
| Success (no drop) | **88 %** | 68 % |
| Drops | **6 / 50** | 16 / 50 |
| Mean slip-recovery latency | **≈ 4 ms** | — |
| In-hand reorientation | **46°** about the vial axis (wrist fixed) | — |
| Determinism | same seed → identical numbers | same |

> The closed loop **cuts the baseline's drops from 16 to 6** (a 20-point success gain) over 50 seeded
> trials. We report **88 % over the full randomization range** honestly rather than narrowing the
> ranges to manufacture a higher number — auditability is the point.

---

## 🧠 How it works

```mermaid
flowchart LR
  M["MuJoCo model<br/>Panda + LEAP + vial/cap + bench"]
  S["Sensors<br/>fingertip touch · wrist force/torque<br/>vial pose · mj_contactForce"]
  D["Slip detector<br/>friction-cone margin<br/>mu*fn - |ft|"]
  F["Sensor-gated FSM"]
  C["Control<br/>Cartesian impedance arm<br/>+ finger grip / escalation"]
  M --> S --> D --> F
  S --> F
  F --> C --> M
  C -. "qfrc_applied + position targets" .-> M
```

The loop closes on **genuine MuJoCo contact forces** (`mj_contactForce`), not a scripted timeline.
The **friction-cone margin** `μ·fₙ − ‖fₜ‖` is the core tactile signal: when it drops toward zero, slip
is imminent.

### Task pipeline (every success transition is sensor-gated)

```mermaid
stateDiagram-v2
  [*] --> APPROACH
  APPROACH --> GRASP: EE at pre-grasp pose
  GRASP --> HOLD: force closure (3+ fingers)
  HOLD --> RECOVER: margin below threshold
  RECOVER --> HOLD: margin restored
  HOLD --> UNCAP: stable hold
  UNCAP --> REORIENT: cap past uncap angle
  REORIENT --> TRANSFER: in-hand yaw at target
  TRANSFER --> SEAL: vial over waste port
  SEAL --> [*]: vial in port, EE retracted
  HOLD --> FAILED: drop, contamination, crush
```

Timeouts exist only as **failure guards** — never as success triggers. Each phase is implemented and
validated as its own deterministic, sensor-gated FSM (`slipzero/fsm.py`).

---

## 🤖 Robot platform

- **Arm:** Franka Emika Panda (7 DOF) · **Hand:** LEAP Hand (16 DOF), dexterous five-finger.
- Composed at load time from MuJoCo Menagerie assets via `MjSpec` (the LEAP palm is attached to the
  Panda flange; the bench/vial/cap/rack/decapper/port scene and all SlipZero sensors are added
  programmatically). See `assets/ATTRIBUTION.md`.

> We evaluated the FFAI starter platforms (Aegis quadruped, Futurist and FF Master humanoids) and
> chose a dedicated 16-DOF dexterous hand: this task is fine in-hand manipulation, which needs
> articulated fingers and a friction-cone tactile signal that the FFAI locomotion platforms
> (wrist + fist, no fingers) do not provide.

---

## 🔬 Technical approach

- **Cartesian impedance arm control** — task-space PD on a hand grasp site with gravity/Coriolis
  compensation, mapped to joint torques through the site Jacobian. Joint-space damping tames the
  7-DOF nullspace; tracks a commanded pose to ≈ 1 mm.
- **Real contact-force sensing** — fingertip↔vial contacts read via `mj_contactForce`, decomposed
  into normal/tangential; per-finger normal force, force-closure, and the friction-cone margin.
- **Slip detection** — the minimum friction-cone margin is low-pass filtered and flagged only after
  it stays below threshold for *k* consecutive steps (rejects contact-force noise).
- **Recovery** — on incipient slip, grip force is escalated to restore the margin; recovery latency
  is logged. A fixed-grip baseline (no recovery) provides the contrast row.
- **MuJoCo depth** — `cone="elliptic"`, `condim=6` (torsional/rolling friction), `implicitfast`
  integrator, hinge-coupled cap, `xfrc_applied` perturbations, four sensor types.

---

## ✅ What it does

| Phase | Capability | Sensor gate |
|---|---|---|
| 0 | Composed Panda+LEAP rig, instrumented | sensors report on contact |
| 1 | Impedance grasp → lift → 5 s upright hold | per-finger force closure |
| 2 | Slip detection + grip-escalation recovery | friction-cone margin |
| 3 | Contact-driven uncap → transfer → seal | cap angle · vial-in-port · contamination |
| 4 | In-hand reorientation (~46°, wrist fixed) | vial yaw vs target |
| 5 | Seeded audit + open-loop baseline + repro test | deterministic metrics |

---

## 🎯 How SlipZero addresses each evaluation criterion

| Criterion | Evidence in this submission | Verify |
|---|---|---|
| **Reproducibility** | Deterministic — same seed → identical numbers. One command per phase; `eval.py` reproduces the headline table; pinned `requirements.txt`; no hardcoded paths; runs from a fresh clone; repro test. | `python eval.py --trials 50 --seed 0` · `pytest tests/test_repro.py` |
| **MuJoCo depth** | `MjSpec` composition (LEAP palm welded to the Panda flange); **four sensor types** (fingertip `touch`, wrist `force`/`torque`, vial `framepos`/`framequat`, `cap_thread` jointpos) **plus `mj_contactForce`**; friction-cone margin from real contact forces; `cone="elliptic"`, `condim=6`, `implicitfast`; hinge-coupled threaded cap; `xfrc_applied` perturbations; arm torque control via `qfrc_applied`. | `slipzero/env.py` · `slipzero/sensors.py` · `assets/slipzero_bench.xml` |
| **Task design** | A clear, hard, real-world "never-drop hazardous sample" workflow — 6 sensor-gated stages with explicit fail conditions (drop, contamination-mat contact, crush) — that stays non-trivial under domain randomization. | pipeline diagram above · `slipzero/fsm.py` |
| **Control** | Cartesian **impedance** control (site Jacobian + gravity/Coriolis comp + nullspace damping, ≈1 mm tracking); event-driven **sensor-gated** FSMs; incipient-slip detection + grip-escalation recovery; open-loop baseline for contrast. | `slipzero/control.py` · `slipzero/fsm.py` |
| **Dexterity** | 16-DOF dexterous hand; multi-finger **force-closure** grasp; **contact-driven cap unscrew**; **in-hand reorientation** of the vial about its axis with the wrist held fixed. | `phase1.py` · `phase3.py` · `phase4.py` |
| **Engineering quality** | Clean modular package (`env` / `control` / `sensors` / `fsm`); every tunable in `config/default.yaml`; pinned deps; Menagerie attribution + MIT license; deterministic; automated tests. | repo layout above · `config/default.yaml` |
| **Presentation** | ≈40 s HD video **produced by the code**, with live telemetry overlays, event stamps, and audit + reproduce cards — every on-screen number sourced from `metrics.csv`. | `render_demo.py` · `demo.mp4` |
| **Innovation** | **Auditable robustness** (every headline number reproduced by one command) + a **tactile closed loop that closes on genuine contact forces** (not a scripted timeline) + in-hand reorient, shown with a measured baseline-beating contrast. | `eval.py` · the audit card in the video |

Guiding principle: **every claim has a matching command and a matching pixel** — the README numbers, the
terminal output of `eval.py`, and the video overlays are the same numbers.

## ▶ Reproduce it (one command each)

Python 3.11. From this folder:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py --headless     # Phase 0: rig + sensors respond to contact
python phase1.py              # grasp → lift → hold
python phase2.py              # slip detection + recovery (baseline vs closed loop)
python phase3.py              # uncap → transfer → seal
python phase4.py              # in-hand reorient
python eval.py --trials 50 --seed 0   # seeded audit → results/metrics.csv + headline table
pytest tests/test_repro.py    # asserts the eval is deterministic

mjpython demo.py              # interactive viewer (macOS); python demo.py elsewhere
python render_demo.py         # regenerate demo.mp4 (needs the ffmpeg binary on PATH)
```

Every check prints a `PHASE n: PASS/FAIL` line. The demo video's audit card reads its numbers from
`metrics.csv`, so the on-screen numbers match `eval.py` exactly.

---

## 🗂 Repository layout

```
submissions/slipzero/
├── slipzero/
│   ├── env.py        # MjSpec composition, sensors, randomization, perturbation
│   ├── control.py    # Cartesian impedance arm controller
│   ├── sensors.py    # mj_contactForce decomposition, force closure, slip detector
│   └── fsm.py        # Phase 1–4 sensor-gated state machines
├── assets/           # vendored Panda + LEAP (Menagerie) + slipzero_bench.xml
├── config/default.yaml   # all tunables (gains, thresholds, randomization)
├── demo.py phase1.py phase2.py phase3.py phase4.py   # per-phase checks
├── eval.py           # seeded N-trial audit → metrics.csv
├── render_demo.py    # telemetry-driven demo video
├── tests/test_repro.py
├── demo.mp4
└── registration.json
```

---

## ⚠️ Limitations (honest)

- The pipeline is tuned around a nominal operating point. It is reliable across a **moderate
  friction/mass envelope** (the eval reports the reliable band) but not the entire randomization
  range — hence **88 %**, reported honestly. Grip escalation helps for downward slide-out slips
  (where the closed loop beats the baseline); a fixed grip is near its physical ceiling for the
  slipperiest, lightest, hardest-hit cases.
- The in-hand reorient gait is open-loop (a tuned finger trajectory) with a **sensor-verified**
  outcome, rather than continuously closed-loop on the vial yaw.
- The uncap success gate uses the cap's absolute angle; the realized rotation (~61°) is larger.
- No real fluid simulation (MuJoCo has none); the "sample" is a rigid vial.

## 🚀 Future improvements

- A margin-proportional, non-ejecting **adaptive grip** to widen the reliable envelope toward the
  full randomization range and push success past 95 % while keeping the closed-loop advantage.
- Continuously **feedback-corrected** in-hand reorient gait.
- `mjx`-parallel evaluation to push the trial count into the hundreds.

---

## 📜 Attribution & license

Project code: MIT (`LICENSE`). Vendored Franka Panda (Apache-2.0) and LEAP Hand (MIT) models are from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie); see `assets/ATTRIBUTION.md`.
