from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from phase1 import _controller
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase2FSM, Phase3FSM, Phase4FSM
from slipzero.sensors import ContactReader

_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
WIDTH, HEIGHT, FPS, STRIDE = 960, 720, 30, 25


def _camera(lookat, distance, azimuth=120, elevation=-18):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.asarray(lookat, float)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _fonts():
    return ImageFont.truetype(_FONT, 34), ImageFont.truetype(_FONT, 22)


def _label(frame, title, subtitle, fonts):
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3]))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, 0, WIDTH, 78], fill=(8, 12, 20, 170))
    draw.text((22, 8), title, font=fonts[0], fill=(255, 255, 255, 255))
    draw.text((24, 48), subtitle, font=fonts[1], fill=(120, 225, 165, 255))
    return np.asarray(image)[:, :, :3]


def _capture(env, fsm, renderer, camera):
    frames = []
    original_step = env.step
    counter = [0]

    def capturing_step(n=1):
        original_step(n)
        counter[0] += 1
        if counter[0] % STRIDE == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render().copy())

    env.step = capturing_step
    fsm.run()
    env.step = original_step
    return frames


def _headline(csv_path):
    rows = list(csv.DictReader(open(csv_path))) if Path(csv_path).exists() else []
    if not rows:
        return None
    n = len(rows)
    closed = 100.0 * sum(int(r["success"]) for r in rows) / n
    base = 100.0 * sum(1 - int(r["baseline_dropped"]) for r in rows) / n
    drops = sum(int(r["dropped"]) for r in rows)
    return {"trials": n, "closed": closed, "base": base, "drops": drops}


def _audit_card(headline, fonts, seconds=2):
    image = Image.new("RGB", (WIDTH, HEIGHT), (10, 14, 22))
    draw = ImageDraw.Draw(image)
    draw.text((40, 60), "Audited robustness", font=fonts[0], fill=(255, 255, 255))
    if headline:
        lines = [
            f"{headline['trials']} seeded randomized trials",
            f"SlipZero (closed loop):  {headline['closed']:.0f}% success   {headline['drops']} drops",
            f"Open-loop baseline:      {headline['base']:.0f}% success",
            "every number reproduced by  python eval.py",
        ]
    else:
        lines = ["run python eval.py to generate metrics.csv"]
    for i, line in enumerate(lines):
        draw.text((44, 150 + i * 46), line, font=fonts[1], fill=(150, 225, 180))
    frame = np.asarray(image)
    return [frame] * (FPS * seconds)


def _encode(frames, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame[:, :, :3]).tobytes())
    proc.stdin.close()
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="SlipZero telemetry-driven demo video")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--csv", default="results/metrics.csv")
    parser.add_argument("--out", default="results/demo.mp4")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    phase1 = dict(config["phase1"]); phase1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    phase2 = dict(config["phase2"]); phase2["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    phase3 = dict(config["phase3"])
    phase4 = dict(config["phase4"])

    env = SlipZeroEnv(arm_torque=True)
    controller = _controller(env, config)
    contacts = ContactReader(env)
    renderer = mujoco.Renderer(env.model, HEIGHT, WIDTH)
    fonts = _fonts()

    near = _camera([0.34, -0.1, 0.62], 0.6, azimuth=110, elevation=-12)
    wide = _camera([0.45, 0.0, 0.5], 1.0, azimuth=120, elevation=-20)

    beats = []

    env.reset()
    fsm = Phase2FSM(env, controller, contacts, phase1, phase2, recovery_enabled=False)
    beats.append((_capture(env, fsm, renderer, near), "SlipZero", "Baseline: fixed grip - the vial slips and drops"))

    env.reset()
    fsm = Phase2FSM(env, controller, contacts, phase1, phase2, recovery_enabled=True)
    beats.append((_capture(env, fsm, renderer, near), "SlipZero", "Closed loop: friction-cone margin triggers grip recovery"))

    env.reset()
    fsm = Phase3FSM(env, controller, contacts, phase1, phase3)
    beats.append((_capture(env, fsm, renderer, wide), "SlipZero", "Grasp - uncap - transfer - seal in the waste port"))

    env.reset()
    fsm = Phase4FSM(env, controller, contacts, phase1, phase4)
    beats.append((_capture(env, fsm, renderer, near), "SlipZero", "In-hand reorientation by finger gait"))

    frames = []
    for beat_frames, title, subtitle in beats:
        for frame in beat_frames:
            frames.append(_label(frame, title, subtitle, fonts))
    frames.extend(_audit_card(_headline(args.csv), fonts))

    code = _encode(frames, args.out)
    duration = len(frames) / FPS
    print(f"frames={len(frames)}  duration={duration:.1f}s  ffmpeg_exit={code}  out={args.out}")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
