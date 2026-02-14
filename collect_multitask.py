"""
Nemotron-VLA: Multi-Task Data Collection
=========================================
Collects expert demonstrations for ALL MetaWorld tasks that have
built-in expert policies. Each task gets a natural language instruction.
"""

import os
import numpy as np
from data_collection import collect_task_demonstrations
from dataset_schema import build_transition_payload, unpack_transition_payload

# ── Natural language instructions for MetaWorld tasks ──
# Maps env_name → human instruction string
TASK_INSTRUCTIONS = {
    "assembly-v3":        "insert the peg into the hole",
    "basketball-v3":      "put the ball into the basket",
    "bin-picking-v3":     "pick up the object from the bin",
    "box-close-v3":       "close the box lid",
    "button-press-v3":    "press the button from the top",
    "button-press-topdown-v3":      "press the button from above",
    "button-press-topdown-wall-v3": "press the button from above near the wall",
    "button-press-wall-v3":         "press the button near the wall",
    "coffee-button-v3":   "press the coffee machine button",
    "coffee-pull-v3":     "pull the coffee mug",
    "coffee-push-v3":     "push the coffee mug",
    "dial-turn-v3":       "turn the dial",
    "disassemble-v3":     "pull the peg out of the hole",
    "door-close-v3":      "close the door",
    "door-lock-v3":       "lock the door",
    "door-open-v3":       "open the door",
    "door-unlock-v3":     "unlock the door",
    "drawer-close-v3":    "close the drawer",
    "drawer-open-v3":     "open the drawer",
    "faucet-close-v3":    "close the faucet",
    "faucet-open-v3":     "open the faucet",
    "hammer-v3":          "hammer the nail",
    "hand-insert-v3":     "insert the object by hand",
    "handle-press-v3":    "press the handle down",
    "handle-press-side-v3":   "press the handle from the side",
    "handle-pull-v3":     "pull the handle",
    "handle-pull-side-v3":    "pull the handle from the side",
    "lever-pull-v3":      "pull the lever",
    "peg-insert-side-v3": "insert the peg from the side",
    "peg-unplug-side-v3": "unplug the peg from the side",
    "pick-out-of-hole-v3":"pick the object out of the hole",
    "pick-place-v3":      "pick up the object and place it at the goal",
    "pick-place-wall-v3": "pick and place the object over the wall",
    "plate-slide-v3":     "slide the plate",
    "plate-slide-back-v3":        "slide the plate back",
    "plate-slide-back-side-v3":   "slide the plate back from the side",
    "plate-slide-side-v3":        "slide the plate from the side",
    "push-v3":            "push the object to the goal",
    "push-back-v3":       "push the object back",
    "push-wall-v3":       "push the object to the goal near the wall",
    "reach-v3":           "reach to the target position",
    "reach-wall-v3":      "reach to the target near the wall",
    "shelf-place-v3":     "place the object on the shelf",
    "soccer-v3":          "kick the soccer ball to the goal",
    "stick-pull-v3":      "use the stick to pull the object",
    "stick-push-v3":      "use the stick to push the object",
    "sweep-v3":           "sweep the object to the goal",
    "sweep-into-v3":      "sweep the object into the hole",
    "window-close-v3":    "close the window",
    "window-open-v3":     "open the window",
}


def get_available_tasks():
    """Return list of tasks that have both an expert policy and an instruction."""
    from metaworld.policies import ENV_POLICY_MAP
    available = []
    for env_name in sorted(ENV_POLICY_MAP.keys()):
        if env_name in TASK_INSTRUCTIONS:
            available.append(env_name)
    print(f"Found {len(available)} tasks with expert policies + instructions")
    return available


def collect_multitask_demonstrations(
    task_list=None,
    camera_name="corner2",
    seed=42,
    episodes_per_task=20,
    max_steps=150,
    output_dir="data/multitask",
):
    """
    Collect expert demonstrations for multiple MetaWorld tasks.

    Args:
        task_list: list of env names, or None for ALL available tasks
        camera_name: MetaWorld camera name
        seed: random seed
        episodes_per_task: number of episodes per task
        max_steps: max steps per episode
        output_dir: directory to save per-task .npz files

    Returns:
        summary: dict with per-task stats
    """
    os.makedirs(output_dir, exist_ok=True)

    if task_list is None:
        task_list = get_available_tasks()

    summary = {}

    for task_idx, env_name in enumerate(task_list):
        instruction = TASK_INSTRUCTIONS.get(env_name, f"complete the {env_name} task")
        print(f"\n{'='*60}")
        print(f"[{task_idx+1}/{len(task_list)}] {env_name}")
        print(f"  Instruction: \"{instruction}\"")
        print(f"{'='*60}")

        save_path = os.path.join(output_dir, f"{env_name}.npz")
        if os.path.exists(save_path):
            print(f"  ⏭️  Already collected, skipping")
            data = np.load(save_path, allow_pickle=True)
            summary[env_name] = {
                "transitions": len(data["states"]),
                "instruction": instruction,
                "skipped": True,
            }
            continue

        try:
            data = collect_task_demonstrations(
                env_name=env_name,
                camera_name=camera_name,
                seed=seed,
                num_episodes=episodes_per_task,
                max_steps=max_steps,
                instruction=instruction,
                verbose=False,
            )
        except Exception as e:
            print(f"  ❌ Failed to collect task: {e}")
            summary[env_name] = {"error": str(e)}
            continue

        np.savez_compressed(
            save_path,
            **build_transition_payload(
                images=data["images"],
                states=data["states"],
                actions=data["actions"],
                instructions=data["instructions"],
                env_names=data["env_names"],
                source="metaworld_expert_multitask",
            ),
        )

        success_rate = data["successes"] / episodes_per_task * 100
        print(
            f"  ✅ {data['transitions']} transitions, "
            f"{data['successes']}/{episodes_per_task} success ({success_rate:.0f}%)"
        )
        print(f"  💾 Saved to {save_path}")

        summary[env_name] = {
            "transitions": data["transitions"],
            "successes": data["successes"],
            "success_rate": success_rate,
            "instruction": instruction,
        }

    # ── Print summary ──
    print(f"\n\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    total_transitions = 0
    for name, info in summary.items():
        if "error" in info:
            print(f"  ❌ {name}: {info['error']}")
        else:
            n = info["transitions"]
            total_transitions += n
            sr = info.get("success_rate", "?")
            print(f"  ✅ {name}: {n} transitions, {sr}% success")

    print(f"\nTotal: {total_transitions} transitions across {len(summary)} tasks")

    return summary


def merge_multitask_datasets(data_dir="data/multitask", output_path="data/multitask_merged.npz"):
    """
    Merge all per-task .npz files into a single dataset.
    Each transition stores its task instruction text.

    Args:
        data_dir: directory with per-task .npz files
        output_path: path for merged dataset

    Returns:
        dict with merged dataset info
    """
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Merging {len(files)} task datasets...")

    all_images, all_states, all_actions = [], [], []
    all_instructions, all_env_names = [], []

    for f in files:
        raw = np.load(f, allow_pickle=True)
        env_name_fallback = os.path.basename(f).replace(".npz", "")
        data = unpack_transition_payload(raw, default_env_name=env_name_fallback)
        n = len(data["states"])
        instruction = str(data["instructions"][0]) if n > 0 else ""
        env_name = str(data["env_names"][0]) if n > 0 else env_name_fallback

        all_images.append(data["images"])
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        all_instructions.extend(data["instructions"].tolist())
        all_env_names.extend(data["env_names"].tolist())

        print(f"  {env_name}: {n} transitions — \"{instruction}\"")

    images = np.concatenate(all_images, axis=0)
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    payload = build_transition_payload(
        images=images,
        states=states,
        actions=actions,
        instructions=np.array(all_instructions, dtype=object),
        env_names=np.array(all_env_names, dtype=object),
        source="metaworld_expert_multitask_merged",
    )
    np.savez_compressed(output_path, **payload)

    print(f"\n💾 Merged dataset saved to {output_path}")
    print(f"   Total transitions: {len(images)}")
    print(f"   Unique tasks: {len(files)}")
    print(f"   Images shape: {images.shape}")

    return {
        "total_transitions": len(images),
        "num_tasks": len(files),
        "output_path": output_path,
        "schema_version": payload["schema_version"],
        "dataset_format": payload["dataset_format"],
    }


# ═══════════════════════════════════════════════════════
# Run standalone
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-task MetaWorld data collection")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--camera", type=str, default="corner2", help="Camera name")
    parser.add_argument("--output-dir", type=str, default="data/multitask")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks (default: all)")
    parser.add_argument("--merge", action="store_true", help="Merge after collection")
    args = parser.parse_args()

    summary = collect_multitask_demonstrations(
        task_list=args.tasks,
        camera_name=args.camera,
        episodes_per_task=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    if args.merge:
        merge_multitask_datasets(data_dir=args.output_dir)
