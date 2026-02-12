"""
Nemotron-VLA: Model Components
===============================
Vision-Language-Action model powered by NVIDIA foundation models:
  - Vision:   NVIDIA RADIO (frozen feature extractor)
  - Language:  NVIDIA Nemotron Nano (frozen text encoder)
  - Action:   Diffusion Policy head (trainable)
  - Fusion:   Cross-attention module (trainable)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. VISION ENCODER — NVIDIA RADIO (frozen)                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_radio_model(device="cuda", dtype=torch.float16):
    """
    Load NVIDIA RADIO vision foundation model via torch.hub.

    RADIO (AM-RADIO) distills knowledge from multiple teacher models
    (CLIP, DINOv2, SAM) into a single efficient ViT backbone.

    Returns:
        model: RADIO model (frozen, eval mode)
        radio_dim: int, dimension of the summary embedding
    """
    print("📸 Loading NVIDIA RADIO vision encoder...")

    # torch.hub is the officially supported way to load RADIO
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version="radio_v2.5-b",   # ViT-B/16 variant, ~86M params
        progress=True,
        skip_validation=True,
    )
    model = model.to(device=device, dtype=dtype).eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Detect output dimension with a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        summary, spatial = model(dummy)
        radio_dim = summary.shape[-1]

    print(f"   ✅ RADIO loaded — summary dim: {radio_dim}")
    return model, radio_dim


def extract_radio_features(radio_model, images_np, device="cuda", batch_size=64):
    """
    Extract RADIO summary embeddings for a batch of images.

    Args:
        radio_model: Loaded RADIO model (from torch.hub)
        images_np: numpy array (N, H, W, 3) uint8
        device: torch device
        batch_size: batch size for processing

    Returns:
        features: numpy array (N, radio_dim) float32
    """
    import torchvision.transforms as T

    # RADIO's preferred resolution and normalization
    # The torch.hub model includes an input_conditioner for normalization
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),  # [0, 1]
    ])

    # Check if model has input conditioner (torch.hub RADIO models do)
    conditioner = getattr(radio_model, "input_conditioner", None)

    N = images_np.shape[0]
    all_features = []

    radio_model.eval()
    dtype = next(radio_model.parameters()).dtype

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_imgs = []
        for i in range(start, end):
            img_t = transform(images_np[i])
            batch_imgs.append(img_t)

        batch_tensor = torch.stack(batch_imgs).to(device=device, dtype=dtype)

        # Apply RADIO's built-in normalization if available
        if conditioner is not None:
            batch_tensor = conditioner(batch_tensor)

        with torch.no_grad():
            summary, _ = radio_model(batch_tensor)

        all_features.append(summary.float().cpu())

        if (start // batch_size) % 10 == 0:
            print(f"   RADIO features: {end}/{N}")

    features = torch.cat(all_features, dim=0).numpy()
    print(f"   ✅ Extracted RADIO features: {features.shape}")
    return features


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. LANGUAGE ENCODER — NVIDIA Nemotron Nano 9B v2 (frozen)      ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_nemotron_model(device="cuda"):
    """
    Load NVIDIA Nemotron Nano 9B v2 for text embedding extraction.

    Uses the hybrid Mamba-2 + Transformer architecture.
    Loaded in float16 for efficient inference on A100.

    Returns:
        model: Nemotron model
        tokenizer: Nemotron tokenizer
        hidden_dim: int, hidden dimension of the model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    print(f"🧠 Loading {model_name} for text encoding...")
    print("   (This may take a few minutes to download ~18GB)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Detect hidden dimension from model config
    hidden_dim = model.config.hidden_size
    print(f"   ✅ Nemotron loaded — hidden dim: {hidden_dim}")

    return model, tokenizer, hidden_dim


def extract_nemotron_embedding(model, tokenizer, text, device="cuda"):
    """
    Extract a text embedding from Nemotron by mean-pooling
    the last hidden layer's representations.

    Args:
        model: Nemotron model
        tokenizer: Nemotron tokenizer
        text: instruction string

    Returns:
        embedding: numpy array (hidden_dim,) float32
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Take the last hidden state and mean-pool over tokens
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            embedding = last_hidden.mean(dim=1)

    embedding = embedding.squeeze(0).float().cpu().numpy()
    print(f"   ✅ Nemotron text embedding: {embedding.shape}")
    return embedding


def unload_model(model):
    """Free GPU memory by deleting model and clearing cache."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("   🗑️  Model unloaded, GPU memory freed")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. STATE ENCODER (trainable)                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class StateEncoder(nn.Module):
    """Encode robot proprioceptive state into a d_model-dim vector."""

    def __init__(self, state_dim: int, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, state):
        """state: (B, state_dim) -> (B, d_model)"""
        return self.ln(self.net(state))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. CROSS-ATTENTION FUSION (trainable)                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class CrossAttentionFusion(nn.Module):
    """
    Fuse vision, language, and state embeddings via cross-attention.

    Upgrade from mini-VLA's simple MLP concatenation:
    - Uses multi-head cross-attention to let modalities attend to each other
    - Followed by a feedforward network for the final fused representation
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Stack vision, language, state as a 3-token sequence
        # then use self-attention to fuse them
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, vis_token, txt_token, state_token):
        """
        Each input: (B, d_model)
        Returns: (B, d_model) — fused context vector
        """
        # Stack into sequence: (B, 3, d_model)
        tokens = torch.stack([vis_token, txt_token, state_token], dim=1)

        # Self-attention over the 3 modality tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Feedforward
        ff_out = self.ff(tokens)
        tokens = self.norm2(tokens + ff_out)

        # Mean-pool over the 3 tokens to get a single fused vector
        fused = tokens.mean(dim=1)  # (B, d_model)
        return fused


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. DIFFUSION POLICY HEAD (trainable)                            ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class DiffusionConfig:
    T: int = 20                 # diffusion timesteps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    action_dim: int = 4         # MetaWorld action dim
    cond_dim: int = 256         # fused context dim


def make_beta_schedule(cfg: DiffusionConfig):
    """Linear beta schedule for DDPM."""
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """t: (B,) integer timesteps -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class ActionDenoiseNetwork(nn.Module):
    """
    Noise prediction network: ε_θ(x_t, t, cond)
    Predicts the noise added to the action at timestep t,
    conditioned on the fused VLA context.
    """

    def __init__(self, cfg: DiffusionConfig, time_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = cfg.action_dim + time_emb_dim + cfg.cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        """
        x_t:  (B, action_dim) noisy action
        t:    (B,) integer timestep
        cond: (B, cond_dim) fused context
        """
        t_emb = self.time_emb(t)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        return self.net(x)


class DiffusionPolicyHead(nn.Module):
    """
    DDPM-based action generation head.
    Learns to denoise random Gaussian noise into robot actions,
    conditioned on the fused vision-language-state context.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoise_net = ActionDenoiseNetwork(cfg)

        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = √(ᾱ_t) * x0 + √(1-ᾱ_t) * ε"""
        ab_t = self.alpha_bar[t].unsqueeze(-1)  # (B, 1)
        return torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise

    def loss(self, actions, cond):
        """
        Training loss: MSE between predicted and true noise.

        actions: (B, action_dim) ground-truth actions
        cond:    (B, cond_dim)   fused context
        """
        B = actions.size(0)
        t = torch.randint(0, self.cfg.T, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)
        eps_pred = self.denoise_net(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond):
        """
        Reverse diffusion: generate actions from noise.

        cond: (B, cond_dim)
        Returns: (B, action_dim)
        """
        B = cond.size(0)
        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)

        for step in reversed(range(self.cfg.T)):
            t = torch.full((B,), step, device=cond.device, dtype=torch.long)
            eps_pred = self.denoise_net(x_t, t, cond)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bar[step]

            # Predict x0, then compute x_{t-1}
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * noise
            else:
                x_t = x0_pred

        return x_t


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. NEMOTRON-VLA — Full VLA Model                               ║
# ╚══════════════════════════════════════════════════════════════════╝

class NemotronVLA(nn.Module):
    """
    Nemotron-VLA: Vision-Language-Action Model

    Architecture:
        ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐
        │ NVIDIA RADIO │   │ NVIDIA Nemotron  │   │ Robot State  │
        │ (frozen ViT) │   │ Nano 9B (frozen) │   │  (raw obs)   │
        └──────┬──────┘   └────────┬─────────┘   └──────┬───────┘
               │                   │                     │
        ┌──────▼──────┐   ┌───────▼────────┐   ┌───────▼───────┐
        │ Vision Proj  │   │  Text Proj     │   │ State Encoder │
        │ Linear→LN   │   │  Linear→LN     │   │  MLP→LN       │
        └──────┬──────┘   └───────┬────────┘   └───────┬───────┘
               │                   │                     │
               └───────────┬───────┴─────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Cross-Attn  │
                    │   Fusion    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Diffusion  │
                    │ Policy Head │
                    └──────┬──────┘
                           │
                      ┌────▼────┐
                      │ Actions │
                      └─────────┘

    During training:  precomputed RADIO & Nemotron embeddings → trainable layers
    During inference: RADIO runs online, Nemotron embedding is cached
    """

    def __init__(
        self,
        radio_dim: int,
        nemotron_dim: int,
        state_dim: int,
        action_dim: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        diffusion_T: int = 20,
    ):
        super().__init__()
        self.d_model = d_model

        # Projection layers for frozen encoder outputs
        self.vision_proj = nn.Sequential(
            nn.Linear(radio_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(nemotron_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Trainable state encoder
        self.state_encoder = StateEncoder(state_dim, d_model)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(d_model, n_heads)

        # Diffusion action head
        diff_cfg = DiffusionConfig(
            T=diffusion_T,
            action_dim=action_dim,
            cond_dim=d_model,
        )
        self.diffusion_head = DiffusionPolicyHead(diff_cfg)

        # Save config for checkpoint
        self.config = {
            "radio_dim": radio_dim,
            "nemotron_dim": nemotron_dim,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "diffusion_T": diffusion_T,
        }

    def _fuse(self, vis_emb, txt_emb, state):
        """Project and fuse all modalities."""
        vis = self.vision_proj(vis_emb)       # (B, d_model)
        txt = self.text_proj(txt_emb)         # (B, d_model)
        st = self.state_encoder(state)        # (B, d_model)
        return self.fusion(vis, txt, st)      # (B, d_model)

    def compute_loss(self, vis_emb, txt_emb, state, actions):
        """
        Training forward pass.

        Args:
            vis_emb:  (B, radio_dim)    precomputed RADIO features
            txt_emb:  (B, nemotron_dim) precomputed Nemotron features
            state:    (B, state_dim)    robot proprioceptive state
            actions:  (B, action_dim)   ground-truth expert actions

        Returns:
            loss: scalar diffusion loss
        """
        cond = self._fuse(vis_emb, txt_emb, state)
        return self.diffusion_head.loss(actions, cond)

    @torch.no_grad()
    def act(self, vis_emb, txt_emb, state):
        """
        Inference: generate action from current observation.

        Args:
            vis_emb: (1, radio_dim)    RADIO features for current image
            txt_emb: (1, nemotron_dim) cached Nemotron text embedding
            state:   (1, state_dim)    current robot state

        Returns:
            action: (1, action_dim)
        """
        self.eval()
        cond = self._fuse(vis_emb, txt_emb, state)
        return self.diffusion_head.sample(cond)


def count_parameters(model):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
