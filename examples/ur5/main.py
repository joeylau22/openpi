"""OpenPI UR5 example – inference/roll‑out driver.

Place this as examples/ur5/main.py inside the OpenPI repository.

Run:
    python examples/ur5/main.py --external_camera base
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import signal
import time
from pathlib import Path

import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image
import tyro
import tqdm

from openpi_client import websocket_client_policy, image_tools
from ur5_sim.env import UR5Env

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    base_camera_topic: str = "/camera/color/image_raw"
    wrist_camera_topic: str = "/wrist_camera/color/image_raw"
    external_camera: str = "base"
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    max_timesteps: int = 600
    open_loop_horizon: int = 8
    save_dir: Path = Path("results")


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    assert args.external_camera in {"base", "wrist"}
    env = UR5Env(args.base_camera_topic, args.wrist_camera_topic)
    policy = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        prompt = input("Enter instruction: ")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_frames = []
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        obs = env.reset()
        bar = tqdm.tqdm(range(args.max_timesteps), desc="Rollout")
        try:
            for t in bar:
                t0 = time.time()
                obs = env.get_observation()
                extern_img = obs[f"{args.external_camera}_rgb"]
                video_frames.append(extern_img)
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    request = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(extern_img, 224, 224),
                        "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224),
                        "observation/joint_position": obs["joints"],
                        "observation/gripper_position": obs["gripper"],
                        "prompt": prompt,
                    }
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy.infer(request)["actions"]
                    assert pred_action_chunk.shape == (10, 8)
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                gripper_open = 1.0 if action[-1] > 0.5 else 0.0
                action = np.concatenate([action[:-1], [gripper_open]])
                action = np.clip(action, -1, 1)
                obs, _, done, _ = env.step(action)
                dt = time.time() - t0
                if dt < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - dt)
                if done:
                    break
        except KeyboardInterrupt:
            print("Rollout interrupted")

        if video_frames:
            save_path = args.save_dir / f"ur5_rollout_{timestamp}.mp4"
            ImageSequenceClip(video_frames, fps=10).write_videofile(str(save_path), codec="libx264")
            print(f"Saved video to {save_path}")

        if input("Run another episode? (y/n) ").lower() != "y":
            break


if __name__ == "__main__":
    tyro.cli(main)