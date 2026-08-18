# Wan-VAE Temporal PCA and Clustering Analysis (2026-08-18)

- Quality: exploratory
- Server: `aliyun-47-116-39-128-1017`
- Branch: `analysis/wan-vae-temporal-web`
- Worktree: `/mnt/home/renqian/oneNFE/FastGen-wan-vae-analysis`
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (`vae` subfolder)
- Dataset: `/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv`
- Dataset snapshot: 10,000 videos (10,001 CSV lines including header)
- Sample selection: 24 rows, shuffled with seed 10
- Frames: zero-based 30, 31, 32, 33, 34 at 256x256
- Encoding: deterministic posterior mode, framewise
- Feature: 4x4 spatial grid pooling of each 48-channel latent frame
- Analysis: standardized absolute and first-frame-relative delta features; SVD PCA;
  k-means++ with 20 restarts and silhouette-based K selection over 2..8
- Output: `/mnt/home/renqian/imgGen/runtime/fastgen/analysis/wan_vae_temporal_web_20260818`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL_FILES_ONLY=true \
HF_HOME=/mnt/home/renqian/imgGen/runtime/MODEL/huggingface \
/mnt/home/renqian/oneNFE/FastGen-tfd/.conda/envs/fastgen/bin/python \
scripts/experiments/analyze_wan_vae_temporal.py \
  --output-dir /mnt/home/renqian/imgGen/runtime/fastgen/analysis/wan_vae_temporal_web_20260818 \
  --num-samples 24 --seed 10 --start-frame 30 --frames 5 \
  --width 256 --height 256 --encode-mode framewise \
  --feature-pool spatial-grid --grid-size 4
```

## Results

Pending development-machine execution.
