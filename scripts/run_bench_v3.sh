#!/usr/bin/env bash
# V3 推理效率基准（A10）。跑前先确认 adapter 与数据都在。
set -e

ADAPTER=/mnt/workspace/runs/v3/adapter
DATA=/mnt/workspace/data/hybrid_v3
OUT=/mnt/workspace/runs/v3

echo "=== 环境自检 ==="
python -c "import torch, unsloth; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
test -d "$ADAPTER" || { echo "缺少 $ADAPTER"; exit 1; }
test -f "$DATA/test.parquet" || { echo "缺少 $DATA/test.parquet"; exit 1; }
mkdir -p "$OUT"

echo "=== 1/3 短请求 + 批量吞吐（370 token, 3 问题 / 9 选项）==="
python -u latency_bench.py \
  --adapter "$ADAPTER" \
  --out "$OUT/latency_a10.json" \
  --max-seq 4500 --max-batch 32

echo "=== 2/3 真实长序列（test 集 p50/p90/max, 整条一次直出）==="
python -u bench_long_v3.py \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --out "$OUT/latency_long_a10.json" \
  --max-seq 4500 --n 32 --max-batch 4

echo "=== 3/3 冷启动 / 前 5 次推理 ==="
python -u cold_latency.py \
  --adapter "$ADAPTER" \
  --out "$OUT/cold_a10.json"

echo "=== 完成 ==="
ls -lh "$OUT"/latency_a10.json "$OUT"/latency_long_a10.json "$OUT"/cold_a10.json
cat "$OUT"/latency_long_a10.json
