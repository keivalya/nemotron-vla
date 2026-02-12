"""
Nemotron-VLA: Environment Wrapper
=================================
MetaWorld MT1 environment wrapper for robot manipulation tasks.
Compatible with Meta-World's gymnasium interface.
"""

import gymnasium as gym
import numpy as np

try:
    import metaworld
except ImportError:
    print("⚠️  metaworld not installed. Run: pip install metaworld")


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
        dict with keys: images, states, actions, instruction
    """
    from metaworld.policies import ENV_POLICY_MAP

    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera_name,
    )

    obs, info = env.reset(seed=seed)
    policy = ENV_POLICY_MAP[env_name]()

    all_images, all_states, all_actions = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = policy.get_action(obs)
            img = env.render()

            all_images.append(img.astype(np.uint8).copy())
            all_states.append(np.asarray(obs, dtype=np.float32).ravel().copy())
            all_actions.append(np.asarray(action, dtype=np.float32).copy())

            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

        success = int(info.get("success", 0))
        print(f"  Episode {ep+1}/{num_episodes}: {steps} steps, success={success}")

    env.close()

    data = {
        "images": np.stack(all_images, axis=0),      # (N, H, W, 3)
        "states": np.stack(all_states, axis=0),       # (N, state_dim)
        "actions": np.stack(all_actions, axis=0),     # (N, action_dim)
        "instruction": instruction,
    }

    print(f"\n✅ Collected {data['images'].shape[0]} transitions")
    print(f"   images:  {data['images'].shape}")
    print(f"   states:  {data['states'].shape}")
    print(f"   actions: {data['actions'].shape}")

    return data
