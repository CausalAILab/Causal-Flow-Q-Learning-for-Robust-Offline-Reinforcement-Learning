"""Load trained agents and visualize their trajectories as high-resolution MP4 videos."""

import json
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.22"

import jax
import flax
import imageio
import ogbench
import numpy as np
from tqdm import trange

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.evaluation import supply_rng

# ── Paths & settings ──────────────────────────────────────────────────────────
CHECKPOINTS = {
    "fql": "/home/ml/explogs/fql/fql/visual-puzzle-4x4-play-singletask-task1-v0/sd1234_20260127_030116/params_1000000.pkl",
    "robust_en_fql": "/home/ml/explogs/fql/fql/visual-puzzle-4x4-play-singletask-task1-v0_offon/sd1234_20260128_103903/params_1000000.pkl",
}

RENDER_WIDTH = 1024
RENDER_HEIGHT = 1024
NUM_EPISODES = 1
VIDEO_FPS = 20
VIDEO_FRAME_SKIP = 1
SEED = 1234


def load_agent(checkpoint_path, env_name):
    """Create an agent from a checkpoint and restore its parameters.

    Reads the companion flags.json to reconstruct the correct architecture,
    then deserialises the saved parameters into it.
    """
    exp_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(exp_dir, "flags.json"), "r") as f:
        flags = json.load(f)

    config = flags["agent"]
    frame_stack = flags.get("frame_stack")

    # Build a low-res env just to get observation / action shapes.
    tmp_env = make_env_and_datasets(env_name, frame_stack=frame_stack, success_timing='post')
    obs, _ = tmp_env[0].reset()
    action_dim = tmp_env[0].action_space.shape[-1]
    ex_obs = obs[np.newaxis]  # add batch dim
    ex_actions = np.zeros((1, action_dim), dtype=np.float32)

    # Create agent skeleton.
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(SEED, ex_obs, ex_actions, config)

    # Restore parameters.
    with open(checkpoint_path, "rb") as f:
        saved = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, saved['agent'])
    print(f"Restored agent from {checkpoint_path}")

    return agent, frame_stack


def rollout_video(agent, env_name, frame_stack, num_episodes=NUM_EPISODES):
    """Run the agent and collect high-resolution rendered frames."""
    # Create a high-resolution render environment.
    _, eval_env, _, _ = make_env_and_datasets(
        env_name, 
        frame_stack=frame_stack, 
        success_timing='post'
    )
    render_env = ogbench.make_env_and_datasets(
        env_name,
        env_only=True,
        width=1024,
        height=1024,
        pixel_transparent_arm=False,
        visualize_info=True,
    )
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(SEED))
    all_frames = []

    for ep in trange(num_episodes, desc="Recording episodes"):
        obs, info = eval_env.reset()
        _, _ = render_env.reset()
        render_env.unwrapped.set_state(info["qpos"], info["qvel"], info["button_states"])
        all_frames.append(render_env.render().copy())
        done = False
        step = 0
        while not done:
            action = actor_fn(observations=obs, temperature=0)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1

            if step % VIDEO_FRAME_SKIP == 0 or done:
                render_env.unwrapped.set_state(info["qpos"], info["qvel"], info["button_states"])
                all_frames.append(render_env.render().copy())

        success = info.get("success", info.get("episode", {}).get("success", "N/A"))
        print(f"  Episode {ep + 1}: {step} steps, success={success}")

    eval_env.close()
    render_env.close()
    return all_frames


def save_video(frames, path, fps=VIDEO_FPS):
    """Write a list of uint8 RGB frames to an MP4 file."""
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264",
        pixelformat="yuv420p",
        output_params=["-s", f"{RENDER_WIDTH}x{RENDER_HEIGHT}", "-crf", "18"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved video ({len(frames)} frames, {len(frames)/fps:.1f}s) → {path}")


if __name__ == "__main__":
    env_name = "visual-puzzle-4x4-play-singletask-task1-v0"
    for label, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {label}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n{'='*60}\nVisualising: {label}\n{'='*60}")
        agent, frame_stack = load_agent(ckpt_path, env_name)
        frames = rollout_video(agent, env_name, frame_stack)
        out_path = f"{label}_{env_name}.mp4"
        save_video(frames, out_path)
