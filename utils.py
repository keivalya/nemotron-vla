"""
Nemotron-VLA: Utilities
=======================
Dataset, training loop, evaluation, and video helpers.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. DATASET                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

class ArrayVLADataset(Dataset):
    """
    Dataset backed by in-memory numpy arrays.

    This keeps the minimal notebook API simple while sharing the same
    data access behavior as file-based datasets.
    """

    def __init__(self, radio_features, nemotron_embeddings, states, actions):
        self.vision_features = np.asarray(radio_features, dtype=np.float32)
        self.text_embeddings = np.asarray(nemotron_embeddings, dtype=np.float32)
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)

        if self.text_embeddings.ndim == 1:
            # Single instruction embedding shared by all samples.
            self._shared_text = True
        elif self.text_embeddings.ndim == 2:
            if len(self.text_embeddings) != len(self.states):
                raise ValueError(
                    f"Mismatch: {len(self.states)} states vs {len(self.text_embeddings)} text embeddings"
                )
            self._shared_text = False
        else:
            raise ValueError("nemotron_embeddings must be rank-1 or rank-2")

        if len(self.states) != len(self.vision_features):
            raise ValueError(
                f"Mismatch: {len(self.states)} states vs {len(self.vision_features)} vision features"
            )
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"Mismatch: {len(self.states)} states vs {len(self.actions)} actions"
            )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        vis = torch.from_numpy(self.vision_features[idx])
        txt_np = self.text_embeddings if self._shared_text else self.text_embeddings[idx]
        txt = torch.from_numpy(np.asarray(txt_np, dtype=np.float32).copy())
        state = torch.from_numpy(self.states[idx])
        action = torch.from_numpy(self.actions[idx])
        return vis, txt, state, action


class NemotronVLADataset(Dataset):
    """
    Dataset for training Nemotron-VLA with precomputed embeddings.

    Loads precomputed RADIO vision features and Nemotron text embedding,
    along with robot states and expert actions.
    """

    def __init__(self, data_path: str, embeddings_path: str):
        """
        Args:
            data_path: path to .npz with images, states, actions, instruction
            embeddings_path: path to .npz with radio_features, nemotron_embedding
        """
        # Load raw demonstration data
        raw = np.load(data_path, allow_pickle=True)
        states = raw["states"].astype(np.float32)    # (N, state_dim)
        actions = raw["actions"].astype(np.float32)   # (N, action_dim)

        # Load precomputed embeddings
        emb = np.load(embeddings_path, allow_pickle=True)
        vision_features = emb["radio_features"].astype(np.float32)  # (N, radio_dim)
        text_embedding = emb["nemotron_embedding"].astype(np.float32)  # (nemotron_dim,)

        self._dataset = ArrayVLADataset(
            radio_features=vision_features,
            nemotron_embeddings=text_embedding,
            states=states,
            actions=actions,
        )

        print(f"📦 Dataset loaded: {len(self)} samples")
        print(f"   vision features: {vision_features.shape}")
        print(f"   text embedding:  {text_embedding.shape}")
        print(f"   states:          {states.shape}")
        print(f"   actions:         {actions.shape}")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. TRAINING                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def train_nemotron_vla(
    model,
    dataset,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    save_path: str = "checkpoints/nemotron_vla.pt",
    log_interval: int = 5,
):
    """
    Train the Nemotron-VLA model.

    Args:
        model: NemotronVLA model
        dataset: NemotronVLADataset
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: weight decay for AdamW
        device: training device
        save_path: path to save checkpoint
        log_interval: print loss every N epochs

    Returns:
        losses: list of average loss per epoch
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )

    losses = []
    best_loss = float("inf")

    print(f"\n🏋️ Training Nemotron-VLA for {epochs} epochs")
    print(f"   batch_size={batch_size}, lr={lr}, samples={len(dataset)}")
    print(f"   batches/epoch={len(loader)}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for vis, txt, state, action in loader:
            vis = vis.to(device)
            txt = txt.to(device)
            state = state.to(device)
            action = action.to(device)

            loss = model.compute_loss(vis, txt, state, action)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * vis.size(0)
            n_samples += vis.size(0)

        scheduler.step()
        avg_loss = total_loss / n_samples
        losses.append(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "epoch": epoch,
                "loss": avg_loss,
            }, save_path)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"loss={avg_loss:.6f} | "
                  f"best={best_loss:.6f} | "
                  f"lr={current_lr:.2e}")

    print("-" * 60)
    print(f"✅ Training complete! Best loss: {best_loss:.6f}")
    print(f"   Checkpoint saved to: {save_path}")

    return losses


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. EVALUATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def evaluate_nemotron_vla(
    model,
    radio_model,
    text_embedding,
    env_name="push-v3",
    camera_name="corner2",
    seed=42,
    num_episodes=5,
    max_steps=150,
    device="cuda",
    save_video=True,
    video_dir="videos",
):
    """
    Evaluate Nemotron-VLA in the MetaWorld environment.

    Args:
        model: trained NemotronVLA
        radio_model: RADIO vision model (frozen)
        text_embedding: precomputed text embedding (nemotron_dim,)
        env_name: MetaWorld task
        num_episodes: number of eval episodes
        max_steps: max steps per episode
        device: torch device
        save_video: whether to save episode videos
        video_dir: directory for videos

    Returns:
        results: dict with rewards and success rates
    """
    import torchvision.transforms as T

    # Ensure rendering backend is configured before env import
    from env import _setup_rendering
    _setup_rendering()
    from env import MetaWorldMT1Wrapper

    # RADIO preprocessing — no manual normalization, model has its own conditioner
    radio_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    radio_dtype = next(radio_model.parameters()).dtype
    conditioner = getattr(radio_model, "input_conditioner", None)

    # Prepare text embedding tensor (reused every step)
    txt_emb = torch.from_numpy(text_embedding).float().unsqueeze(0).to(device)

    # Create environment
    env = MetaWorldMT1Wrapper(
        env_name=env_name,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera_name,
    )

    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        import imageio

    model.eval()
    radio_model.eval()

    all_rewards = []
    all_successes = []

    print(f"\n🎯 Evaluating Nemotron-VLA on {env_name}")
    print(f"   episodes={num_episodes}, max_steps={max_steps}")
    print("-" * 60)

    for ep in range(num_episodes):
        img, state, info = env.reset()
        ep_reward = 0.0
        frames = [img.copy()]
        success = False

        for step in range(max_steps):
            # Extract RADIO features for current image
            img_t = radio_transform(img).unsqueeze(0).to(device=device, dtype=radio_dtype)
            if conditioner is not None:
                img_t = conditioner(img_t)
            with torch.no_grad():
                summary, _ = radio_model(img_t)
                vis_emb = summary.float()  # (1, radio_dim)

            # Prepare state
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Generate action
            with torch.no_grad():
                action_t = model.act(vis_emb, txt_emb, state_t)

            action_np = action_t.squeeze(0).cpu().numpy()

            # Step environment
            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            frames.append(img.copy())

            if int(info.get("success", 0)) == 1:
                success = True

            if done:
                break

        all_rewards.append(ep_reward)
        all_successes.append(success)

        status = "✅ SUCCESS" if success else "❌ FAIL"
        print(f"   Episode {ep+1}/{num_episodes}: "
              f"reward={ep_reward:.2f}, steps={step+1}, {status}")

        # Save video
        if save_video:
            video_path = os.path.join(video_dir, f"{env_name}_ep{ep+1:03d}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)

    env.close()

    results = {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "success_rate": np.mean(all_successes),
        "rewards": all_rewards,
        "successes": all_successes,
    }

    print("-" * 60)
    print(f"📊 Results:")
    print(f"   Mean reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Success rate: {results['success_rate']*100:.1f}%")

    return results


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. CHECKPOINT LOADING                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_checkpoint(checkpoint_path, device="cuda"):
    """Load a trained NemotronVLA from checkpoint."""
    from models import NemotronVLA

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = NemotronVLA(
        radio_dim=cfg["radio_dim"],
        nemotron_dim=cfg["nemotron_dim"],
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        diffusion_T=cfg["diffusion_T"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"   Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.6f}")

    return model


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. VISUALIZATION                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def plot_training_loss(losses, save_path=None):
    """Plot training loss curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2, color="#76b900")  # NVIDIA green
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Diffusion Loss", fontsize=12)
    ax.set_title("Nemotron-VLA Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   📈 Loss plot saved to {save_path}")

    plt.show()
    return fig


def display_sample_frames(data_path, n_frames=8):
    """Display sample frames from collected demonstrations."""
    import matplotlib.pyplot as plt

    data = np.load(data_path, allow_pickle=True)
    images = data["images"]
    N = len(images)

    indices = np.linspace(0, N - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(2.5 * n_frames, 2.5))
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(f"t={idx}", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("Expert Demonstration Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig
