"""
Compatibility helpers for the minimal notebook.

This module intentionally re-exports core functionality from `models.py`,
`utils.py`, and `env.py` so there is a single implementation of model,
dataset, and training logic across the repository.
"""

import os
import numpy as np
import torch

from env import MetaWorldMT1Wrapper
from models import (
    DiffusionConfig,
    NemotronVLA,
    extract_nemotron_embedding,
    extract_radio_features,
    load_nemotron_model,
    load_radio_model,
    unload_model,
)
from utils import ArrayVLADataset, train_nemotron_vla


# Backward-compatible aliases used in nemotron_vla_minimal.ipynb
load_radio = load_radio_model
load_nemotron = load_nemotron_model
unload = unload_model
VLADataset = ArrayVLADataset


def train(
    model,
    dataset,
    epochs=80,
    batch_size=128,
    lr=3e-4,
    device="cuda",
    save_path="nemotron_vla.pt",
):
    """Compatibility wrapper over utils.train_nemotron_vla."""
    return train_nemotron_vla(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_path=save_path,
        log_interval=10,
    )


def make_env(env_name="push-v3", camera="corner2", seed=42):
    """Return a raw Meta-World MT1 gymnasium environment."""
    import gymnasium as gym
    import metaworld  # noqa: F401

    return gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera,
    )


def run_inference(
    vla_model,
    radio_model,
    text_emb_np,
    env_name,
    instruction,
    max_steps=150,
    camera="corner2",
    seed=42,
    video_path=None,
    device="cuda",
):
    """Run one episode and optionally save a rollout video."""
    import imageio
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )
    conditioner = getattr(radio_model, "input_conditioner", None)
    radio_dtype = next(radio_model.parameters()).dtype

    txt = torch.from_numpy(text_emb_np).float().unsqueeze(0).to(device)

    env = MetaWorldMT1Wrapper(
        env_name=env_name,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera,
    )
    img, state, _ = env.reset(seed=seed)

    frames = [img.copy()]
    reward_sum = 0.0
    success = False

    vla_model.eval()
    radio_model.eval()

    for step in range(max_steps):
        img_t = transform(img).unsqueeze(0).to(device=device, dtype=radio_dtype)
        if conditioner is not None:
            img_t = conditioner(img_t)

        with torch.no_grad():
            vis, _ = radio_model(img_t)
            st = torch.from_numpy(state).float().unsqueeze(0).to(device)
            act = vla_model.act(vis.float(), txt, st).squeeze(0).cpu().numpy()

        img, state, reward, done, info = env.step(act)
        reward_sum += reward
        frames.append(img.copy())

        if int(info.get("success", 0)) == 1:
            success = True
        if done:
            break

    env.close()

    if video_path is None:
        os.makedirs("inference", exist_ok=True)
        clean_instruction = instruction.replace(" ", "_")[:30]
        video_path = f"inference/{env_name}_{clean_instruction}.mp4"

    with imageio.get_writer(video_path, fps=20) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))

    status = "SUCCESS" if success else "FAIL"
    print(f"  {status} | reward={reward_sum:.1f} | steps={step + 1} | {video_path}")
    return {
        "reward": reward_sum,
        "success": success,
        "steps": step + 1,
        "video_path": video_path,
    }
