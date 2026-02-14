"""
Shared data collection helpers for Meta-World expert demonstrations.
"""

import numpy as np
from dataset_schema import build_transition_payload


def collect_task_demonstrations(
    env_name="push-v3",
    camera_name="corner2",
    seed=42,
    num_episodes=100,
    max_steps=150,
    instruction=None,
    verbose=True,
):
    """
    Collect expert-policy demonstrations for one Meta-World task.

    Returns:
        dict with keys:
          images, states, actions, instruction, env_name,
          transitions, successes, episodes
    """
    import gymnasium as gym
    import metaworld  # noqa: F401
    from metaworld.policies import ENV_POLICY_MAP

    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera_name,
    )
    policy = ENV_POLICY_MAP[env_name]()
    env.reset(seed=seed)

    all_images, all_states, all_actions = [], [], []
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = policy.get_action(obs)
            img = np.flipud(env.render()).copy()  # fix OpenGL orientation

            all_images.append(img.astype(np.uint8))
            all_states.append(np.asarray(obs, dtype=np.float32).ravel().copy())
            all_actions.append(np.asarray(action, dtype=np.float32).copy())

            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

        ep_success = int(info.get("success", 0)) == 1
        if ep_success:
            successes += 1
        if verbose:
            print(
                f"  Episode {ep + 1}/{num_episodes}: {steps} steps, success={int(ep_success)}"
            )

    env.close()

    if not all_states:
        raise RuntimeError(
            "No transitions were collected. Increase max_steps or verify environment setup."
        )

    instruction_text = instruction if instruction is not None else ""
    data = build_transition_payload(
        images=np.stack(all_images, axis=0),
        states=np.stack(all_states, axis=0),
        actions=np.stack(all_actions, axis=0),
        instructions=instruction_text,
        env_names=env_name,
        source="metaworld_expert",
    )
    data["transitions"] = int(data["num_transitions"])
    data["successes"] = int(successes)
    data["episodes"] = int(num_episodes)
    return data
