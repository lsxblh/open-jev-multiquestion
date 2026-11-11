#!/usr/bin/env bash
# v3 冒烟：20 步，确认数据 + max_seq 4500 能跑通
set -e
cd /mnt/workspace
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python -c "from unsloth import FastDecisionModel;print('unsloth OK')"
ls -d models/Qwen3.5-4B data/hybrid_v3/train.parquet
python -u train_decision.py \
  --data /mnt/workspace/data/hybrid_v3 \
  --model /mnt/workspace/models/Qwen3.5-4B \
  --out runs/v3_smoke \
  --steps 20 --max-seq 4500 --bs 2 --accum 8 --lr 2e-4 \
  --eval-every 10 --eval-rows 32 --skip-baseline
echo '=== 冒烟完成，检查 [data] train: 15580 items, skipped 0/... ==='
