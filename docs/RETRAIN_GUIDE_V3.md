# v3 重训包 · 2026-11-06

## 这份数据怎么来的（如实记录，不粉饰）

规则由你逐条确认，`max_seq = 4500`：

```
train:
  排除 painting 的 14,403 个原生 case → 原样保留（1 state + 多问题，不拆不合并）
  painting 的 138 个超长 case → 拆成 23,552 个单问题片段
                              → 按 (case 尺寸, label) 分层随机抽 5% = 1,177 条
                              → seed 3407，稀有 label 保底 1 条
  = 15,580 case / 55,523 题

calibration / validation / test / ood:
  排除 painting，其余原样保留，不拆、不合并、不重采样
```

**为什么 max_seq 定 4500**：`security_incidents` 实测最长 4,396，4500 刚好够，
这样 11 个 source 全部整条保留，只有 painting（20,233~60,212 token）装不下。

## 评测集被动过的地方（诚实标注）

排除 painting 是 `max_seq=4500` 下的物理约束，不是为了让数字好看。
**这些题在评测里不出现**，写报告时请如实说明：

| split | 排除的 painting case | 排除的题 | 占原切分 |
|---|---:|---:|---:|
| calibration | 3 | 512 | 11.1% |
| validation | 3 | 512 | 14.0% |
| test | 18 | 3,072 | 30.1% |
| ood | 18 | 6,912 | 44.5% |

排除后各切分：

| split | case | 题数 | N>1 case | groups |
|---|---:|---:|---:|---:|
| train | 15,580 | 55,523 | 9,765 | 7,256 |
| calibration | 1,078 | 4,086 | 707 | 467 |
| validation | 932 | 3,145 | 588 | 451 |
| test | 1,902 | 7,136 | 1,224 | 887 |
| ood | 1,772 | 8,627 | 1,772 | 1,396 |

⚠️ 因此本轮 `test accuracy` 与上一轮 97.66% **口径不同，不可比**
（上一轮是全拆版、每行 1 问题；这一轮是原生多问题整条评测，
且评测集排除了 painting）。不要拿两个数直接说「提升/下降」。

## 本地已验证

```
跨 split group 泄漏: 无
5 个切分抽样 max 长度 4,394 ≤ 4500，>4500 的 0 条
decision_config max_len = 4500
build_dataset: 5 个切分全部 skipped 0
```

## 开机后

```bash
cd /mnt/workspace && unzip -o jev_retrain_v3.zip
```

若环境被重置（上一轮发生过）：

```bash
pip install -qqq unsloth "transformers==5.17.0" "trl==1.15.0" "tokenizers==0.23.3"
python -c "import torch;print(torch.__version__,torch.cuda.is_available())"
python -c "from unsloth import FastDecisionModel;print('unsloth OK')"
```

`transformers 5.17.0` 要求 `tokenizers>=0.23.1,<0.24.0`，所以钉 0.23.3。
pip 报 ms-swift / vllm / mcore-bridge 的冲突警告是魔搭预装包，不用管。

### ① 冒烟（20 步，约 5–10 分钟）

```bash
bash run_retrain_v3.sh
```

盯 `[data] train: 15580 items, skipped 0/55523`

### ② 正式训练

```bash
mkdir -p runs
nohup python -u train_decision.py \
  --data /mnt/workspace/data/hybrid_v3 \
  --model /mnt/workspace/models/Qwen3.5-4B \
  --out runs/v3 \
  --steps 400 --max-seq 4500 --bs 2 --accum 8 --lr 2e-4 \
  --eval-every 100 --save-every 100 --eval-rows 0 \
  > runs/v3_train.log 2>&1 &
tail -f runs/v3_train.log
```

- `--eval-rows 0` = **全量评测**（test 1,902 case / ood 1,772 case），数字才真实；
  冒烟时才用 32
- `--out runs/v3` 是新目录，**不要加 `--resume`**
- 若 `nvidia-smi` 显存 < 15 GB，下一轮可把 `--bs` 提到 4

### ③ 报告数字（不设通过/不通过）

训练和评测**全部一次直出**，没有逐条再合并这一步。

```bash
# 分语料 / 分题型
python -u eval_breakdown.py \
  --adapter /mnt/workspace/runs/v3/adapter \
  --data /mnt/workspace/data/hybrid_v3 \
  --out /mnt/workspace/runs/v3/breakdown.json

# 推理延时 p50 / p90 / p95 / p99 / 吞吐 req/s
python -u latency_bench.py \
  --adapter /mnt/workspace/runs/v3/adapter \
  --out /mnt/workspace/runs/v3/latency_a10.json \
  --max-seq 4500 --max-batch 32

# 冷启动 + 首次编译
python -u cold_latency.py \
  --adapter /mnt/workspace/runs/v3/adapter \
  --out /mnt/workspace/runs/v3/cold_a10.json
```

训练耗时从这两处取：
- `runs/v3/run_report.json` 的 `train_runtime_s`
- `runs/v3/metrics_log.jsonl` 的 `elapsed_s`（每 100 步的累计）

`latency_bench.py` 默认 `n=32` 次采样，p50/p95 用线性插值分位数；
`--max-seq 4500` 必须传，否则会按 1024 把长序列截断。

## 包内文件

```
data/hybrid_v3/*.parquet       5 切分
data/hybrid_v3/manifest.json   生成规则与统计
train_decision.py              训练 + 标定 + 全量评测 + 保存
eval_breakdown.py              分语料 / 分题型
predict.py                     一次直出推理
latency_bench.py / cold_latency.py  延时基准
run_retrain_v3.sh              一键冒烟
RETRAIN_GUIDE_V3.md            本文件
```

按你的要求：**不含**逐条拆分再合并的脚本（`decide_fields.py` /
`verify_multiquestion.py`），本轮没有这一步。
