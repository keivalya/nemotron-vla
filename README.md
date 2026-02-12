<div align="center">

# Nemotron-VLA

**Vision-Language-Action Model powered by NVIDIA Foundation Models**

*NVIDIA RADIO · NVIDIA Nemotron Nano 9B v2 · Diffusion Policy*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

</div>

---

## Overview

Nemotron-VLA is a fully-functional Vision-Language-Action model that uses NVIDIA's foundation models as its backbone. It demonstrates how to build a VLA that takes camera images + text instructions → robot actions, using state-of-the-art NVIDIA models for vision and language understanding.

### Architecture

```
  NVIDIA RADIO (ViT)     Nemotron Nano 9B v2     Robot State
  [frozen, 0.4-0.7B]     [frozen, 9B]            [raw obs]
         │                       │                     │
    Vision Proj            Text Proj           State Encoder
         │                       │                     │
         └───────────┬───────────┘─────────────────────┘
                     │
            Cross-Attention Fusion
                     │
           Diffusion Policy Head (DDPM)
                     │
                Robot Actions
```

### Models Used

| Modality | Model | Source | Trainable? |
|----------|-------|--------|------------|
| Vision | **NVIDIA RADIO** | [HuggingFace](https://huggingface.co/nvidia/RADIO) | Frozen |
| Language | **NVIDIA Nemotron Nano 9B v2** | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | Frozen |
| Fusion | Cross-Attention Module | Custom | **Trained** |
| Action | Diffusion Policy Head | Custom | **Trained** |

## Getting Started

### Option 1: Google Colab (Recommended)

1. Upload `nemotron_vla.ipynb` to Google Colab
2. Set runtime to **GPU → A100** (Colab Pro required)
3. Run all cells sequentially

The notebook will:
- Install all dependencies
- Write helper modules (`env.py`, `models.py`, `utils.py`)
- Collect expert demonstrations from MetaWorld
- Precompute embeddings from NVIDIA models
- Train the VLA
- Evaluate and save videos

### Option 2: Local Setup

```bash
# Clone and setup
git clone <repo-url>
cd nemotron-vla

# Install dependencies
pip install torch torchvision transformers accelerate
pip install gymnasium metaworld mujoco
pip install causal-conv1d mamba-ssm
pip install imageio[ffmpeg] matplotlib
```

## Project Structure

```
nemotron-vla/
├── nemotron_vla.ipynb    # Main Colab notebook (run this!)
├── env.py                # MetaWorld environment wrapper
├── models.py             # All model components
│   ├── RADIO loading     #   NVIDIA RADIO vision encoder
│   ├── Nemotron loading  #   Nemotron Nano 9B text encoder
│   ├── StateEncoder      #   Robot state MLP
│   ├── CrossAttentionFusion  # Multi-modal fusion
│   ├── DiffusionPolicyHead   # DDPM action generation
│   └── NemotronVLA       #   Full VLA model
├── utils.py              # Dataset, training, evaluation
└── README.md             # This file
```

## Memory Management Strategy

The key insight enabling training on A100 40GB:

1. **Load RADIO** → extract vision features for all images → **unload** (~1.4GB freed)
2. **Load Nemotron 9B** → extract text embedding → **unload** (~18GB freed)
3. **Train** only lightweight fusion + diffusion (~0.8M params, <4GB)

## Key Differences from mini-VLA

| Feature | mini-VLA | Nemotron-VLA |
|---------|----------|--------------|
| Vision | TinyCNN (random init) | NVIDIA RADIO (pretrained) |
| Language | GRU + SimpleTokenizer | Nemotron Nano 9B v2 |
| Fusion | MLP concatenation | Cross-attention |
| Action Head | 2-layer MLP denoiser | 3-layer with LayerNorm |
| Training | End-to-end | Precompute + train head |

## License

This project uses NVIDIA models under their respective licenses:
- NVIDIA RADIO: [License](https://huggingface.co/nvidia/RADIO)
- NVIDIA Nemotron: [NVIDIA Open Model License](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
