# Installation Guide

This project is designed for high-memory remote GPUs (A100/H100/H200 class), including Google Colab and HPC clusters.

## 1. Prerequisites

- Python 3.10+ recommended
- NVIDIA GPU with sufficient VRAM (A100 40GB or better strongly recommended)
- CUDA-compatible PyTorch install
- MuJoCo-compatible runtime libraries (EGL or OSMesa)

## 2. Clone the Repository

```bash
git clone <repo-url>
cd nemotron-vla
```

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If your platform requires a specific CUDA wheel for PyTorch, install `torch`/`torchvision` first using your platform command, then run:

```bash
pip install -r requirements.txt --no-deps
```

## 4. Platform-Specific Notes

### Google Colab

- Use a GPU runtime.
- Prefer A100 runtime when available.
- Run `nemotron_vla.ipynb` from top to bottom.

### HPC / Cluster (Explorer, Slurm, etc.)

- Load your CUDA module first (example):

```bash
module load cuda
```

- Create and activate a virtual environment or conda environment.
- Install dependencies with `pip install -r requirements.txt`.
- Ensure rendering backend libraries are discoverable (`libEGL.so` preferred).

### Local Linux Workstation

- Install NVIDIA drivers and CUDA toolkit matching your PyTorch build.
- Ensure MuJoCo can use EGL (headless) or OSMesa fallback.

## 5. Quick Environment Validation

Run this in the project root:

```bash
python3 -m py_compile env.py models.py utils.py helpers.py collect_multitask.py data_collection.py dataset_schema.py
```

Optionally verify imports:

```bash
python3 - <<'PY'
import torch, gymnasium, mujoco, transformers, metaworld
print('imports ok')
PY
```

## 6. First Run Paths

- Main workflow: `nemotron_vla.ipynb`
- Simpler workflow: `nemotron_vla_minimal.ipynb`
- CLI multi-task collection: `collect_multitask.py`

## 7. Common Issues

- `MUJOCO_GL` / rendering errors:
  - `env.py` auto-selects EGL/OSMesa, but your system still needs the underlying libraries installed.
- Out-of-memory during model load:
  - Use A100/H200 class GPUs.
  - Precompute vision/text embeddings and unload backbone models before training.
- Nemotron load failures:
  - Recheck `causal-conv1d` and `mamba-ssm` installation.

## 8. Dataset Schema

Collected datasets now use a unified schema with these core fields:

- `schema_version`
- `dataset_format`
- `source`
- `num_transitions`
- `images`
- `states`
- `actions`
- `instructions` (per-transition)
- `env_names` (per-transition)

Legacy fields (`instruction`, `env_name`) are still written when a file contains one unique task/instruction.
