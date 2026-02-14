"""
Nemotron-VLA Helpers by @keivalya
All model components, data loading, training, and inference in one file.
  Vision:  NVIDIA RADIO (frozen)
  Language: NVIDIA Nemotron Nano 9B v2 (frozen)
  Action:  Diffusion Policy (trained)
"""

import math, os, gc, io
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NVIDIA RADIO — Vision Encoder (frozen)
def load_radio(device="cuda"):
    model = torch.hub.load("NVlabs/RADIO", "radio_model", version="radio_v2.5-b",
                           progress=True, skip_validation=True)
    model = model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        dim = model(torch.randn(1, 3, 224, 224, device=device))[0].shape[-1]
    print(f"📸 RADIO loaded — dim={dim}")
    return model, dim


def extract_radio_features(model, images_np, device="cuda", batch_size=64):
    import torchvision.transforms as T
    transform = T.Compose([T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                           T.CenterCrop(224), T.ToTensor()])
    cond = getattr(model, "input_conditioner", None)
    feats = []
    for i in range(0, len(images_np), batch_size):
        batch = torch.stack([transform(img) for img in images_np[i:i+batch_size]]).to(device)
        if cond: batch = cond(batch)
        with torch.no_grad():
            feats.append(model(batch)[0].float().cpu())
        if (i // batch_size) % 20 == 0:
            print(f"   RADIO: {min(i+batch_size, len(images_np))}/{len(images_np)}")
    out = torch.cat(feats).numpy()
    print(f"   RADIO features: {out.shape}")
    return out


# NVIDIA Nemotron Nano 9B v2 — Language Encoder (frozen)
def load_nemotron(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    print(f"Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                  trust_remote_code=True, device_map="auto")
    model.eval()
    dim = model.config.hidden_size
    print(f"   Nemotron loaded — dim={dim}")
    return model, tok, dim


def extract_nemotron_embedding(model, tok, text, device="cuda"):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True)
    dev = next(model.parameters()).device
    ids = inputs["input_ids"].to(dev)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(dev)
    with torch.no_grad():
        h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1)
        else:
            emb = h.mean(1)
    return emb.squeeze(0).float().cpu().numpy()


def unload(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


# NemotronVLA Model
@dataclass
class DiffusionConfig:
    T: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02
    action_dim: int = 4
    cond_dim: int = 256


class NemotronVLA(nn.Module):
    def __init__(self, radio_dim, nemotron_dim, state_dim, action_dim=4, d_model=256, n_heads=4, diffusion_T=20):
        super().__init__()
        self.d_model = d_model
        self.vision_proj = nn.Sequential(nn.Linear(radio_dim, d_model), nn.LayerNorm(d_model))
        self.text_proj = nn.Sequential(nn.Linear(nemotron_dim, d_model), nn.LayerNorm(d_model))
        self.state_enc = nn.Sequential(nn.Linear(state_dim, 128), nn.GELU(), nn.Linear(128, d_model), nn.LayerNorm(d_model))

        # Cross-attention fusion
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

        # Diffusion head
        cfg = DiffusionConfig(T=diffusion_T, action_dim=action_dim, cond_dim=d_model)
        self.diff = DiffusionHead(cfg)
        self.config = dict(radio_dim=radio_dim, nemotron_dim=nemotron_dim, state_dim=state_dim,
                           action_dim=action_dim, d_model=d_model, n_heads=n_heads, diffusion_T=diffusion_T)

    def _fuse(self, vis, txt, state):
        v, t, s = self.vision_proj(vis), self.text_proj(txt), self.state_enc(state)
        tokens = torch.stack([v, t, s], dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + out)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens.mean(dim=1)

    def compute_loss(self, vis, txt, state, actions):
        return self.diff.loss(actions, self._fuse(vis, txt, state))

    @torch.no_grad()
    def act(self, vis, txt, state):
        self.eval()
        return self.diff.sample(self._fuse(vis, txt, state))


class DiffusionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, 0))

        # Time embedding
        self.time_dim = 64
        # Denoiser
        in_dim = cfg.action_dim + self.time_dim + cfg.cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, cfg.action_dim),
        )

    def _time_emb(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def loss(self, actions, cond):
        B = actions.size(0)
        t = torch.randint(0, self.cfg.T, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        ab = self.alpha_bar[t].unsqueeze(-1)
        x_t = torch.sqrt(ab) * actions + torch.sqrt(1 - ab) * noise
        pred = self.net(torch.cat([x_t, self._time_emb(t), cond], -1))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, cond):
        B = cond.size(0)
        x = torch.randn(B, self.cfg.action_dim, device=cond.device)
        for s in reversed(range(self.cfg.T)):
            t = torch.full((B,), s, device=cond.device, dtype=torch.long)
            eps = self.net(torch.cat([x, self._time_emb(t), cond], -1))
            x0 = (x - torch.sqrt(1 - self.alpha_bar[s]) * eps) / torch.sqrt(self.alpha_bar[s])
            x = torch.sqrt(self.alphas[s]) * x0 + torch.sqrt(self.betas[s]) * torch.randn_like(x) if s > 0 else x0
        return x


# Dataset
class VLADataset(Dataset):
    def __init__(self, radio_feats, nemotron_embs, states, actions):
        self.v = radio_feats.astype(np.float32)
        self.t = nemotron_embs.astype(np.float32)
        self.s = states.astype(np.float32)
        self.a = actions.astype(np.float32)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return torch.from_numpy(self.v[i]), torch.from_numpy(self.t[i]), \
               torch.from_numpy(self.s[i]), torch.from_numpy(self.a[i])


# Training
def train(model, dataset, epochs=80, batch_size=128, lr=3e-4, device="cuda", save_path="nemotron_vla.pt"):
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    best = float("inf")
    losses = []
    for ep in range(epochs):
        total, n = 0.0, 0
        for v, t, s, a in loader:
            v, t, s, a = v.to(device), t.to(device), s.to(device), a.to(device)
            loss = model.compute_loss(v, t, s, a)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * v.size(0); n += v.size(0)
        sched.step()
        avg = total / n; losses.append(avg)
        if avg < best:
            best = avg
            torch.save({"model_state_dict": model.state_dict(), "config": model.config, "epoch": ep, "loss": avg}, save_path)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{epochs} | loss={avg:.6f} | best={best:.6f}")
    print(f"Training done. Best loss: {best:.6f}")
    return losses


# Environment
def make_env(env_name="push-v3", camera="corner2", seed=42):
    import gymnasium as gym
    import metaworld  # noqa
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed,
                   render_mode="rgb_array", camera_name=camera)
    return env


# Inference
def run_inference(vla_model, radio_model, text_emb_np, env_name, instruction,
                  max_steps=150, camera="corner2", seed=42, video_path=None, device="cuda"):
    """Run one episode and save video."""
    import torchvision.transforms as T
    import imageio

    transform = T.Compose([T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                           T.CenterCrop(224), T.ToTensor()])
    cond = getattr(radio_model, "input_conditioner", None)
    txt = torch.from_numpy(text_emb_np).float().unsqueeze(0).to(device)

    env = make_env(env_name, camera, seed)
    obs, info = env.reset(seed=seed)
    state = np.asarray(obs, dtype=np.float32).ravel()
    img = np.flipud(env.render()).copy()
    frames = [img]
    reward_sum, success = 0.0, False

    vla_model.eval(); radio_model.eval()
    for step in range(max_steps):
        img_t = transform(img).unsqueeze(0).to(device)
        if cond: img_t = cond(img_t)
        with torch.no_grad():
            vis = radio_model(img_t)[0].float()
            st = torch.from_numpy(state).float().unsqueeze(0).to(device)
            act = vla_model.act(vis, txt, st).squeeze(0).cpu().numpy()
        obs, reward, trunc, term, info = env.step(act)
        state = np.asarray(obs, dtype=np.float32).ravel()
        img = np.flipud(env.render()).copy()
        frames.append(img)
        reward_sum += reward
        if int(info.get("success", 0)) == 1: success = True
        if trunc or term: break
    env.close()

    if video_path is None:
        os.makedirs("inference", exist_ok=True)
        video_path = f"inference/{env_name}_{instruction.replace(' ','_')[:30]}.mp4"
    with imageio.get_writer(video_path, fps=20) as w:
        for f in frames: w.append_data(f)

    status = "SUCCESS" if success else "FAIL"
    print(f"  {status} | reward={reward_sum:.1f} | steps={step+1} | {video_path}")
    return {"reward": reward_sum, "success": success, "steps": step+1, "video_path": video_path}
