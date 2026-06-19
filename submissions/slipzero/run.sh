#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== Phase 0: rig + sensors =="          ; python demo.py --headless
echo "== Phase 1: grasp -> lift -> hold =="  ; python phase1.py
echo "== Phase 2: slip detection + recovery ="; python phase2.py --no-render
echo "== Phase 3: uncap -> transfer -> seal ="; python phase3.py --no-render
echo "== Phase 4: in-hand reorient =="       ; python phase4.py --no-render
echo "== Vision: tactile + dual-camera fusion ="; python vision_check.py
echo "== Phase 5: seeded audit =="           ; python eval.py --trials 50 --seed 0
echo "== Repro test =="                      ; pytest tests/test_repro.py -q
echo "== Submission validator =="            ; python validate_submission.py
echo "== All checks passed. Render the demo with: python render_demo.py =="
