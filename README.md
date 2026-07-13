# DDPM on CIFAR-10

A PyTorch replication of *Denoising Diffusion Probabilistic Models* (Ho, Jain & Abbeel, 2020), trained on CIFAR-10 and packaged as a single-GPU, Colab-friendly pipeline.

**Note on scope:** this is a replication of the DDPM framework, not a byte-exact reproduction of the original paper's setup. The core diffusion formulation and U-Net-style architecture follow the paper, but the pipeline also incorporates several improvements introduced in later work: v-prediction, a cosine noise schedule, self-conditioning, min-SNR-γ loss weighting, EMA, and DDIM sampling for faster inference. If you're benchmarking against the original paper's reported FID, keep this in mind.

## Pipeline

- `ddpm_model.py` — architecture, noise schedule, DDIM sampler
- `ddpm_train.py` — training loop (single-process, checkpoint auto-resume)
- `ddpm_generate.py` — sampling from a checkpoint
- `ddpm_evaluate.py` — FID evaluation via `torch-fidelity`

## Features

- v-prediction parameterization
- Cosine noise schedule
- Self-conditioning
- Min-SNR-γ loss weighting
- EMA weights
- DDIM sampling
- FID evaluation

## Requirements

- Single GPU (T4 or better), Colab or local
- `torch`, `torchvision`, `torch-fidelity`, `tqdm`

## Usage

### 1. Install dependencies

```bash
pip install -q torch-fidelity tqdm
```

### 2. (Optional) Mount Google Drive

Recommended for Colab so checkpoints survive a runtime disconnect. Set `USE_DRIVE = True` in the notebook, or point `CHECKPOINT_DIR` at a persistent path if running locally.

### 3. Train

```bash
python ddpm_train.py --checkpoint_dir checkpoints --epochs 1000 --batch_size 64
```

Training is fully resumable — re-running the command picks up from the latest checkpoint automatically. Lower `--epochs`, `--batch_size`, or `--save_every` for faster iteration.

**A note on scale:** this is the original CIFAR-10 architecture (128–512 channels, attention at 16×16/8×8, batch size 64) trained for up to 1000 epochs. On a single free-tier T4 this is a multi-day job.

### 4. Generate samples

```bash
python ddpm_generate.py --epoch 50 --checkpoint_dir checkpoints --output_dir samples_epoch_50 --total_samples 64 --ddim_steps 500
```

Uses EMA weights and DDIM sampling by default.

### 5. Evaluate FID

FID evaluation requires 10,000 generated images:

```bash
python ddpm_generate.py --epoch 50 --checkpoint_dir checkpoints --output_dir fake_images_epoch_50 --total_samples 10000 --ddim_steps 500
python ddpm_evaluate.py 50
```

Omit the epoch argument to scan and evaluate every `fake_images_epoch_*` folder in the current directory.

## Reference

Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
