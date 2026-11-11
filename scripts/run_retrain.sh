#!/usr/bin/env bash
# 冒烟：验证 hybrid 数据 + max_seq 4096 能跑通（约 5-10 分钟）
set -e
cd /mnt/workspace
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python -c "from unsloth import FastDecisionModel;print('unsloth OK')"
ls -d models/Qwen3.5-4B data/hybrid/train.parquet
python -u train_decision.py \
  --data /mnt/workspace/data/hybrid \
  --model /mnt/workspace/models/Qwen3.5-4B \
  --out runs/hybrid_smoke \
  --steps 20 --max-seq 4096 --bs 2 --accum 4 --lr 2e-4 \
  --eval-every 10 --eval-rows 32 --skip-baseline
echo '=== 冒烟完成，检查 [data] train: ... skipped 0/... ==='

