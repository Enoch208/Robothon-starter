from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from slipzero.control import ArmImpedance
from slipzero.env import SlipZeroEnv
from slipzero.fsm import Phase1FSM
from slipzero.sensors import ContactReader


def _load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    phase = dict(config["phase1"])
    phase["f_grasp_min"] = float(config["thresholds"]["f_grasp_min"])
    return config, phase


def _controller(env, config):
    controller = config["controller"]
    return ArmImpedance(
        env,
        controller["arm_kp_pos"],
        controller["arm_kp_rot"],
        controller["arm_kd_pos"],
        controller["arm_kd_rot"],
        joint_damping=controller["arm_joint_damping"],
    )


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def save_png(path, env, render_config):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = env.sensor("vial_pos")
    camera.distance = render_config["distance"]
    camera.azimuth = render_config["azimuth"]
    camera.elevation = render_config["elevation"]
    width = int(render_config["width"])
    height = int(render_config["height"])
    renderer = mujoco.Renderer(env.model, height, width)
    renderer.update_scene(env.data, camera=camera)
    image = np.ascontiguousarray(renderer.render()[:, :, :3])
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def run(config_path, render=True):
    config, phase = _load_config(config_path)
    env = SlipZeroEnv(arm_torque=True)
    fsm = Phase1FSM(env, _controller(env, config), ContactReader(env), phase)
    result = fsm.run()
    if render:
        save_png(phase["render_path"], env, phase["render"])
    return result, phase["render_path"] if render else None


def main():
    parser = argparse.ArgumentParser(description="SlipZero Phase 1 grasp/lift/hold check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    result, render_path = run(args.config, render=not args.no_render)
    metrics = result.metrics
    print(f"lifted height: {metrics.lifted_height:.3f} m")
    print(f"minimum hold lift: {metrics.min_hold_lift:.3f} m")
    print(f"vial tilt: {metrics.tilt_deg:.1f} deg  max hold tilt: {metrics.max_hold_tilt_deg:.1f} deg")
    print(f"fingers engaged: {metrics.fingers_engaged}  force closure: {metrics.force_closure}")
    if render_path is not None:
        print(f"render: {render_path}")
    print("PHASE 1:", "PASS" if result.passed else f"FAIL ({result.reason})")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
