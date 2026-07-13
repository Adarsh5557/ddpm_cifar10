DDPM on CIFAR-10
A PyTorch replication of Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020), trained on CIFAR-10 and packaged as a single-GPU, Colab-friendly pipeline.
Note on scope: this is a replication of the DDPM framework, not a byte-exact reproduction of the original paper's setup. The core diffusion formulation and U-Net-style architecture follow the paper, but the pipeline also incorporates several improvements introduced in later work: v-prediction, a cosine noise schedule, self-conditioning, min-SNR-γ loss weighting, EMA, and DDIM sampling for faster inference. If you're benchmarking against the original paper's reported FID, keep this in mind.
Pipeline:

ddpm_model.py — architecture, noise schedule, DDIM sampler
ddpm_train.py — training loop (single-process, checkpoint auto-resume)
ddpm_generate.py — sampling from a checkpoint
ddpm_evaluate.py — FID evaluation via torch-fidelity
