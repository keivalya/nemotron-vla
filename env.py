"""
Nemotron-VLA: Environment Wrapper
=================================
MetaWorld MT1 environment wrapper for robot manipulation tasks.
Compatible with Meta-World's gymnasium interface.
"""

import os

def _setup_rendering():
    """
    Auto-detect the best MuJoCo rendering backend.
    - EGL:    hardware-accelerated, works on NVIDIA GPUs (Colab, clusters)
    - OSMesa: software fallback, needs libOSMesa.so

    Handles Colab's quirk where only versioned .so.1 files exist
    (not the bare .so symlinks that ctypes defaults to).
    """
    # If user already set it, respect their choice
    if "MUJOCO_GL" in os.environ:
        return

    import ctypes, ctypes.util

    def _try_load(names):
        for name in names:
            try:
                ctypes.cdll.LoadLibrary(name)
                return True
            except OSError:
                continue
        # Also try ctypes.util.find_library
        for name in names:
            bare = name.replace("lib", "", 1).split(".so")[0]
            found = ctypes.util.find_library(bare)
            if found:
                try:
                    ctypes.cdll.LoadLibrary(found)
                    return True
                except OSError:
                    continue
        return False

    # Try EGL first (preferred on NVIDIA GPUs)
    if _try_load(["libEGL.so", "libEGL.so.1", "libEGL.so.1.1.0"]):
        os.environ["MUJOCO_GL"] = "egl"
        return

    # Try OSMesa as fallback
    if _try_load(["libOSMesa.so", "libOSMesa.so.8", "libOSMesa.so.6"]):
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return

    # Default to EGL and hope MuJoCo resolves it
    os.environ["MUJOCO_GL"] = "egl"

_setup_rendering()

import numpy as np
from data_collection import collect_task_demonstrations


class MetaWorldMT1Wrapper:
    """
    Wraps a MetaWorld MT1 environment into a simple interface:
      - reset() -> (image, state, info)
      - step(action) -> (image, state, reward, done, info)
    """

    def __init__(
        self,
        env_name="push-v3",
        seed=42,
        render_mode="rgb_array",
        camera_name="corner2",
    ):
        import gymnasium as gym
        import metaworld  # noqa: F401 — registers Meta-World envs

        self.env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            camera_name=camera_name,
        )
        self.render_mode = render_mode
        self.env_name = env_name

        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """Extract flat state vector from observation."""
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                parts = []
                if "robot_state" in obs:
                    parts.append(obs["robot_state"])
                if "object_state" in obs:
                    parts.append(obs["object_state"])
                state = np.concatenate(parts, axis=-1)
            else:
                raise KeyError(f"Unknown obs keys: {list(obs.keys())}")
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        """Render and return RGB image."""
        img = self.env.render()
        img = np.flipud(img).copy()
        return img.astype(np.uint8)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()


def collect_demonstrations(
    env_name="push-v3",
    camera_name="corner2",
    seed=42,
    num_episodes=100,
    max_steps=150,
    instruction="push the object to the goal",
):
    """
    Collect expert demonstrations using MetaWorld's built-in policies.

    Returns:
        dict with transition arrays + schema metadata.
        Legacy keys (`instruction`, `env_name`) are preserved for compatibility.
    """
    collected = collect_task_demonstrations(
        env_name=env_name,
        camera_name=camera_name,
        seed=seed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        instruction=instruction,
        verbose=True,
    )

    data = {
        "images": collected["images"],       # (N, H, W, 3)
        "states": collected["states"],       # (N, state_dim)
        "actions": collected["actions"],     # (N, action_dim)
        "instructions": collected["instructions"],  # per-transition instruction text
        "env_names": collected["env_names"],        # per-transition env names
        "schema_version": collected["schema_version"],
        "dataset_format": collected["dataset_format"],
        "source": collected["source"],
        "num_transitions": collected["num_transitions"],
        "instruction": collected["instruction"],
        "env_name": collected["env_name"],
    }

    print(f"\n✅ Collected {data['images'].shape[0]} transitions")
    print(f"   images:  {data['images'].shape}")
    print(f"   states:  {data['states'].shape}")
    print(f"   actions: {data['actions'].shape}")

    return data
