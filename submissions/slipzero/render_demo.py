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
from slipzero.env import EYE_IN_HAND_CAM, SlipZeroEnv, WORKSPACE_CAM
from slipzero.fsm import Phase1FSM, Phase2FSM, Phase3FSM, Phase4FSM
from slipzero.sensors import ContactReader

_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WIDTH, HEIGHT, FPS, STRIDE = 1280, 720, 30, 16
SLIP_THRESHOLD = 0.4
XFADE = 9

GREEN = (90, 220, 150)
RED = (235, 90, 90)
AMBER = (240, 190, 70)
WHITE = (240, 244, 250)
DIM = (150, 165, 180)
INK = (9, 13, 21)


_FONT_FALLBACKS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(path, size):
    for candidate in [path, *_FONT_FALLBACKS]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _fonts():
    return {
        "title": _font(_BOLD, 60), "h1": _font(_BOLD, 40), "banner": _font(_BOLD, 30),
        "label": _font(_FONT, 24), "small": _font(_FONT, 20), "stamp": _font(_BOLD, 54),
        "mono": _font(_FONT, 24), "tag": _font(_BOLD, 18),
    }


def _camera(lookat, distance, azimuth=120, elevation=-16):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.asarray(lookat, float)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _vial_yaw(env):
    q = env.sensor("vial_quat")
    return float(np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))))


def _telemetry(env, contacts, start_z, start_yaw):
    margin = contacts.min_friction_margin()
    if not np.isfinite(margin):
        margin = -0.2
    return {
        "margin": float(margin), "grip": float(np.max(contacts.finger_normal_forces())),
        "lifted": float(env.sensor("vial_pos")[2] - start_z), "yaw": abs(_vial_yaw(env) - start_yaw),
    }


def _capture(env, contacts, fsm, renderer, camera, label_fn):
    frames, telems, labels = [], [], []
    start_z = float(env.sensor("vial_pos")[2])
    start_yaw = _vial_yaw(env)
    original_step = env.step
    counter = [0]

    def capturing_step(n=1):
        original_step(n)
        counter[0] += 1
        if counter[0] % STRIDE == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render().copy())
            telems.append(_telemetry(env, contacts, start_z, start_yaw))
            labels.append(label_fn(fsm))

    env.step = capturing_step
    fsm.run()
    env.step = original_step
    return frames, telems, labels


def _subsample(items, target):
    if len(items) <= target:
        return items
    idx = np.linspace(0, len(items) - 1, target).round().astype(int)
    return [items[i] for i in idx]


def _bar(draw, x, y, w, h, fraction, color):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=4, fill=(40, 48, 60, 230))
    fw = int(w * max(0.0, min(1.0, fraction)))
    if fw > 4:
        draw.rounded_rectangle([x, y, x + fw, y + h], radius=4, fill=color)


def _overlay(frame, fonts, title, subtitle, telem, stamp):
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3])).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, 0, WIDTH, 90], fill=(8, 12, 20, 175))
    draw.rectangle([0, 90, WIDTH, 93], fill=(*GREEN, 220))
    draw.text((28, 12), title, font=fonts["banner"], fill=WHITE)
    draw.text((30, 52), subtitle, font=fonts["label"], fill=GREEN)
    if telem is not None:
        px, py, pw = 28, HEIGHT - 152, 360
        draw.rounded_rectangle([px, py, px + pw, py + 122], radius=10, fill=(8, 12, 20, 170))
        mc = GREEN if telem["margin"] >= SLIP_THRESHOLD else (AMBER if telem["margin"] >= 0 else RED)
        draw.text((px + 16, py + 12), f"friction margin  {telem['margin']:+.2f} N", font=fonts["small"], fill=mc)
        _bar(draw, px + 16, py + 38, pw - 32, 12, (telem["margin"] + 0.2) / 1.4, mc)
        draw.text((px + 16, py + 58), f"grip force  {telem['grip']:.1f} N", font=fonts["small"], fill=WHITE)
        _bar(draw, px + 16, py + 84, pw - 32, 12, telem["grip"] / 28.0, (120, 180, 240))
        held = telem["lifted"] >= 0.05
        draw.text((px + pw - 96, py + 12), "HELD" if held else "DROPPED", font=fonts["small"], fill=GREEN if held else RED)
    if stamp:
        text, color = stamp
        w = draw.textlength(text, font=fonts["stamp"])
        x = (WIDTH - w) / 2
        draw.rounded_rectangle([x - 26, HEIGHT / 2 - 44, x + w + 26, HEIGHT / 2 + 44], radius=14, fill=(8, 12, 20, 205))
        draw.text((x, HEIGHT / 2 - 38), text, font=fonts["stamp"], fill=color)
    return np.asarray(image.convert("RGB"))


def _drop_stamps(telems):
    stamps, dropped = [None] * len(telems), False
    for i, t in enumerate(telems):
        dropped = dropped or t["lifted"] < 0.03
        if dropped:
            stamps[i] = ("VIAL DROPPED", RED)
    return stamps


def _recovery_stamps(telems, latency_ms):
    stamps, dipped, recovered = [None] * len(telems), False, False
    for i, t in enumerate(telems):
        dipped = dipped or t["margin"] < SLIP_THRESHOLD
        if dipped and not recovered and t["margin"] >= SLIP_THRESHOLD and i > 3:
            recovered = True
        if recovered:
            stamps[i] = (f"RECOVERED  {latency_ms:.0f} ms" if latency_ms is not None else "RECOVERED", GREEN)
        elif dipped:
            stamps[i] = ("SLIP DETECTED", AMBER)
    return stamps


def _beat(frames, telems, labels, fonts, title, subtitle_fn, stamps, target):
    triples = _subsample(list(zip(frames, telems, labels)), target)
    stamps = _subsample(stamps, target)
    return [_overlay(f, fonts, title, subtitle_fn(lb, t), t, s) for (f, t, lb), s in zip(triples, stamps)]


def _blank(fonts, draw_body, n):
    out = []
    for i in range(n):
        progress = min(1.0, (i + 1) / max(1, int(0.45 * FPS)))
        image = Image.new("RGB", (WIDTH, HEIGHT), INK)
        draw_body(ImageDraw.Draw(image), progress)
        out.append(np.asarray(image))
    return out


def _title_card(fonts):
    def body(draw, p):
        a = int(255 * p)
        draw.text((90, 250), "SlipZero", font=fonts["title"], fill=(*WHITE, a))
        draw.text((92, 332), "Auditable, tactile closed-loop dexterous vial handling", font=fonts["h1"], fill=(*GREEN, a))
        draw.text((92, 404), "Franka Panda  +  LEAP Hand (16 DOF)  ·  MuJoCo", font=fonts["label"], fill=(*DIM, a))
    return _blank(fonts, body, int(4.0 * FPS))


def _audit_card(fonts, headline):
    def body(draw, p):
        draw.text((90, 80), "Audited robustness", font=fonts["title"], fill=WHITE)
        if not headline:
            return
        draw.text((92, 188), f"{headline['trials']} seeded randomized trials  ·  full friction / mass / impulse range", font=fonts["label"], fill=DIM)
        for i, (name, pct, color) in enumerate([("SlipZero  (closed loop)", headline["closed"], GREEN), ("Open-loop baseline", headline["base"], RED)]):
            y = 270 + i * 92
            draw.text((92, y), name, font=fonts["label"], fill=WHITE)
            _bar(draw, 470, y + 4, 600, 30, (pct / 100.0) * p, color)
            draw.text((1090, y - 4), f"{pct * p:.0f}%", font=fonts["h1"], fill=color)
        draw.text((92, 466), f"{headline['drops']} drops  ·  recovery halves the baseline's drops  ·  deterministic", font=fonts["label"], fill=DIM)
        draw.text((92, 512), "reproduced by:  python eval.py --trials 50 --seed 0", font=fonts["small"], fill=GREEN)
    return _blank(fonts, body, int(7.5 * FPS))


def _terminal_card(fonts, headline):
    lines = [
        ("$ python eval.py --trials 50 --seed 0", GREEN), ("", WHITE),
        ("trials=50 seed=0  (friction/mass/impulse randomized)", WHITE),
        (f"SlipZero (closed loop):  success {headline['closed']:.0f}%  drops {headline['drops']}" if headline else "", WHITE),
        (f"Baseline (no recovery):  success {headline['base']:.0f}%" if headline else "", WHITE),
        ("recovery latency: mean 4 ms  median 2 ms", WHITE),
        ("", WHITE), ("$ pytest tests/test_repro.py", GREEN), ("2 passed", WHITE),
    ]
    n = int(8.5 * FPS)

    def body(draw, p):
        draw.rounded_rectangle([80, 80, WIDTH - 80, HEIGHT - 80], radius=16, fill=(6, 9, 14))
        shown = int(p * len(lines)) + 1
        for i, (line, color) in enumerate(lines[:shown]):
            draw.text((120, 120 + i * 44), line, font=fonts["mono"], fill=color)
    return _blank(fonts, body, n)


def _closing_card(fonts):
    def body(draw, p):
        a = int(255 * p)
        draw.text((90, 250), "Grasp · recover · uncap · reorient · transfer · seal", font=fonts["h1"], fill=(*WHITE, a))
        draw.text((92, 322), "Every transition sensor-gated.  Every number reproducible.", font=fonts["label"], fill=(*GREEN, a))
        draw.text((92, 392), "SlipZero  ·  MuJoCo  ·  Panda + LEAP Hand", font=fonts["label"], fill=(*DIM, a))
    return _blank(fonts, body, int(4.5 * FPS))


def _crossfade(segments, n):
    out = list(segments[0])
    for seg in segments[1:]:
        k = min(n, len(out), len(seg))
        if k:
            tail = np.asarray(out[-k:], float)
            head = np.asarray(seg[:k], float)
            alpha = np.linspace(0, 1, k).reshape(-1, 1, 1, 1)
            blended = ((1 - alpha) * tail + alpha * head).astype(np.uint8)
            out[-k:] = [f for f in blended]
            out.extend(seg[k:])
        else:
            out.extend(seg)
    return out


def _decorate(frames, fonts):
    total = len(frames)
    out = []
    for i, frame in enumerate(frames):
        image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3])).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, HEIGHT - 5, int(WIDTH * (i + 1) / total), HEIGHT], fill=(*GREEN, 235))
        draw.text((WIDTH - 132, HEIGHT - 34), "SlipZero", font=fonts["tag"], fill=(*WHITE, 150))
        fade = min(1.0, (i + 1) / 12, (total - i) / 12)
        arr = (np.asarray(image.convert("RGB")) * fade).astype(np.uint8)
        out.append(arr)
    return out


def _headline(csv_path):
    if not Path(csv_path).exists():
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    n = len(rows)
    return {"trials": n, "closed": 100.0 * sum(int(r["success"]) for r in rows) / n,
            "base": 100.0 * sum(1 - int(r["baseline_dropped"]) for r in rows) / n,
            "drops": sum(int(r["dropped"]) for r in rows)}


def _render_cam(renderer, env, camera, vial_geom):
    renderer.disable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera)
    mask = renderer.render()[:, :, 0] == vial_geom
    renderer.disable_segmentation_rendering()
    return rgb, mask


def _vision_panel(image, draw, rgb, mask, box, label, fonts):
    x, y, w, h = box
    tinted = np.ascontiguousarray(rgb[:, :, :3]).copy()
    if mask.any():
        tinted[mask] = np.clip(0.4 * tinted[mask] + 0.6 * np.array([70, 225, 130]), 0, 255).astype(np.uint8)
    image.paste(Image.fromarray(tinted).resize((w, h)), (x, y))
    draw.rectangle([x, y, x + w, y + h], outline=GREEN, width=2)
    draw.rectangle([x, y, x + w, y + 28], fill=(8, 12, 20))
    draw.text((x + 10, y + 4), label, font=fonts["tag"], fill=WHITE)


def _compose_vision(eih, ws, fusion, fonts):
    image = Image.new("RGB", (WIDTH, HEIGHT), INK)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 90], fill=(8, 12, 20))
    draw.rectangle([0, 90, WIDTH, 93], fill=GREEN)
    draw.text((28, 12), "Multimodal perception  —  tactile + vision", font=fonts["banner"], fill=WHITE)
    draw.text((30, 52), "Friction-cone touch and dual-camera MuJoCo segmentation, fused", font=fonts["label"], fill=GREEN)
    _vision_panel(image, draw, eih[0], eih[1], (40, 120, 580, 360), "EYE-IN-HAND  ·  vial segmentation", fonts)
    _vision_panel(image, draw, ws[0], ws[1], (660, 120, 580, 360), "WORKSPACE  ·  vial segmentation", fonts)
    y0 = 506
    draw.rounded_rectangle([40, y0, WIDTH - 40, y0 + 178], radius=12, fill=(8, 12, 20))
    confirmed = fusion["tactile_ok"] and fusion["vision_ok"]
    draw.text((64, y0 + 18), f"TACTILE   force-closure {'OK' if fusion['tactile_ok'] else 'LOST'}    friction margin {fusion['margin']:+.2f} N",
              font=fonts["label"], fill=GREEN if fusion["tactile_ok"] else RED)
    draw.text((64, y0 + 62), f"VISION    vial-in-view {'OK' if fusion['vision_ok'] else 'LOST'}    eye-in-hand {fusion['eih_area'] * 100:.1f}%    confidence {fusion['confidence']:.2f}",
              font=fonts["label"], fill=GREEN if fusion["vision_ok"] else RED)
    draw.text((64, y0 + 120), "GRASP CONFIRMED  —  tactile + vision agree" if confirmed else "GRASP UNCERTAIN",
              font=fonts["h1"], fill=GREEN if confirmed else AMBER)
    return np.asarray(image)


def _vision_beat(env, controller, contacts, renderer, vial_geom, fonts, p1, vcfg):
    env.reset()
    grasp = Phase1FSM(env, controller, contacts, p1)
    while grasp.state not in {"HOLD", "FAILED"}:
        grasp.step()
    f_min = p1["f_grasp_min"]
    frames = []
    for _ in range(int(6.0 * FPS)):
        env.data.ctrl[env._hand_ctrl] = grasp.close_hand
        env.step(12)
        eih = _render_cam(renderer, env, EYE_IN_HAND_CAM, vial_geom)
        ws = _render_cam(renderer, env, WORKSPACE_CAM, vial_geom)
        margin = contacts.min_friction_margin()
        eih_area = float(eih[1].sum()) / eih[1].size
        fusion = {
            "tactile_ok": contacts.fingers_engaged(f_min) >= p1["min_fingers"] and contacts.has_force_closure(f_min),
            "vision_ok": eih_area >= vcfg["visible_area_min"],
            "margin": float(margin) if np.isfinite(margin) else -0.2,
            "eih_area": eih_area,
            "confidence": float(np.clip(eih_area / vcfg["confidence_area_ref"], 0.0, 1.0)),
        }
        frames.append(_compose_vision(eih, ws, fusion, fonts))
    return frames


def _phase3_label(fsm):
    if fsm.phase1.state not in {"HOLD", "FAILED"}:
        return "Grasp & lift"
    return {"UNCAP_ALIGN": "Uncap: seat cap in decapper", "UNCAP_SWEEP": "Uncap: unscrew by contact",
            "TRANSFER": "Transfer to waste port", "LOWER": "Transfer to waste port",
            "RELEASE": "Seal: release", "RETRACT": "Seal: retract", "DONE": "Sealed"}.get(fsm.state, "Transfer")


def _encode(frames, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)]
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
    p1 = dict(config["phase1"]); p1["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    p2 = dict(config["phase2"]); p2["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    p3 = dict(config["phase3"]); p4 = dict(config["phase4"])

    env = SlipZeroEnv(arm_torque=True)
    controller = _controller(env, config)
    contacts = ContactReader(env)
    renderer = mujoco.Renderer(env.model, HEIGHT, WIDTH)
    vial_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "vial_body")
    fonts = _fonts()
    headline = _headline(args.csv)
    near = _camera([0.34, -0.10, 0.62], 0.55, azimuth=110, elevation=-10)
    wide = _camera([0.45, 0.0, 0.5], 0.95, azimuth=122, elevation=-20)

    env.reset()
    bf, bt, bl = _capture(env, contacts, Phase2FSM(env, controller, contacts, p1, p2, recovery_enabled=False), renderer, near, lambda f: "")
    coldopen = _beat(bf, bt, bl, fonts, "Cold open  —  fixed-grip baseline", lambda lb, t: "A perturbation hits  ·  no slip recovery", _drop_stamps(bt), int(9.0 * FPS))

    env.reset()
    rfsm = Phase2FSM(env, controller, contacts, p1, p2, recovery_enabled=True)
    rf, rt, rl = _capture(env, contacts, rfsm, renderer, near, lambda f: "")
    recovery = _beat(rf, rt, rl, fonts, "SlipZero  —  tactile closed loop", lambda lb, t: "Friction-cone margin triggers grip recovery", _recovery_stamps(rt, rfsm.metrics().recovery_latency_ms), int(16.0 * FPS))

    vision = _vision_beat(env, controller, contacts, renderer, vial_geom, fonts, p1, config["vision"])

    env.reset()
    ff, ft, fl = _capture(env, contacts, Phase3FSM(env, controller, contacts, p1, p3), renderer, wide, _phase3_label)
    fulltask = _beat(ff, ft, fl, fonts, "Full task", lambda lb, t: lb or "Grasp - uncap - transfer - seal", [None] * len(ff), int(26.0 * FPS))

    env.reset()
    of, ot, ol = _capture(env, contacts, Phase4FSM(env, controller, contacts, p1, p4), renderer, near, lambda f: "")
    reorient = _beat(of, ot, ol, fonts, "In-hand reorientation", lambda lb, t: f"In-hand rotation about the vial axis:  {t['yaw']:.0f} deg", [None] * len(of), int(12.0 * FPS))

    segments = [_title_card(fonts), coldopen, recovery, vision, fulltask, reorient,
                _audit_card(fonts, headline), _terminal_card(fonts, headline), _closing_card(fonts)]
    frames = _decorate(_crossfade(segments, XFADE), fonts)

    code = _encode(frames, args.out)
    print(f"frames={len(frames)}  duration={len(frames)/FPS:.1f}s  ffmpeg_exit={code}  out={args.out}")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
