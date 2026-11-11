# 多问题联合决策模型 · Qwen3.5-4B + NF4 + LoRA

> **一句话**：把「一次判一题」改成「一次判多题」。
>
> 一个 state 连同它的多个问题，一次前向全部打分，直接输出**校准过的概率**，不生成文本、不需要解析 JSON。

判别式决策模型：给定一段状态描述和一组带候选的问题（`choice` / `score` / `noul`），
模型一次前向给出每个候选的概率分布。基于 [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev)
与 [Unsloth FastDecisionModel](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)。

---

## 为什么做这个

上一版把每个 case 全拆成「1 state + 1 question」。但决策头里让多个 field 互相参照的
self-attention，**训练时从未见过多个问题共存** —— 联合决策能力根本没有被训练出来。

所以这一轮没有调参，而是回到数据：逐条审读 12 类语料的任务说明，判定哪些能拆、哪些不能拆。

| 裁决 | 依据 |
|---|---|
| **不能拆** | 4 个 workflow 语料原文写有 `Exactly one final disposition is selected` / `The output is an action set`，多个问题是一次互斥决策，拆开即丢约束 |
| 本来就 1 题 | `tic_tac_toe` / `tile_platformer` / `trex_runner` |
| **可以拆** | 只有 `painting-geometry`（一条 20k–60k token），每题自包含 |

把 `max_seq` 从 1024 提到 **4500**（实测最长 `security_incidents` = 4,396 token），
11 个数据源得以整条保留；只有 painting 的 138 个训练 case 拆成片段后按
`(case 尺寸, label)` 分层抽样 5% = 1,177 条，避免它独占算力。

**训练 / 评测 / 推理全部「一次直出」，没有逐条再合并这一步。**

逐条审查的原文证据见 [`results/split_legality.md`](results/split_legality.md)，
生成逻辑见 [`scripts/make_hybrid_v3.py`](scripts/make_hybrid_v3.py)。

---

## 主要结果

**主指标是整条全对率（`record_accuracy`）**：一个 state 下所有问题全部答对才算对，
逐题 accuracy 会掩盖这个能力。

| | 整条全对率 | 逐题 accuracy | ECE |
|---|---:|---:|---:|
| 训练前基线（validation） | 0.1062 | 0.4560 | 0.0472 |
| step 400（validation） | 0.6470 | 0.8811 | 0.0253 |
| **test** | **0.6593** | **0.8960** | **0.0134** |
| **ood** | 0.6919 | 0.9220 | 0.0161 |

分语料看，4 类 workflow 稳定在 **0.97–0.99**，其中 `security_incidents`
（78 个 case / 1,326 题，每 case 17 题）逐题平均 0.9864 —— 不拆联合约束的直接成效。

**推理效率（A10，FA2 on）**：短请求 p50 **0.175 s**、p95 0.178 s；
batch=32 时 17.02 req/s；4,384 token / 17 题的长场景 p50 **0.709 s**、p95 0.714 s。

训练：A10 22 GB，400 步 / 有效 batch 16，**59.5 分钟**，`epoch = 0.4108`。
可训练参数 64.51M / 3.07B = **2.10%**（LoRA r=16 + 从零初始化的决策头）。

完整曲线与分语料表：`results/v3_run_report.json`、`results/v3_breakdown_test.txt`、
`results/v3_latency_long_a10.json`。

---

## 快速开始：直接跑一次推理

权重已用 Git LFS 放在 `model/adapter_v3/`，clone 后可直接演示。

```bash
git lfs install && git lfs pull

pip install unsloth "transformers==5.17.0" "trl==1.15.0" "tokenizers==0.23.3"

python scripts/predict.py --adapter model/adapter_v3 --demo
```

首次运行会按 `adapter_config.json` 里的 `Qwen/Qwen3.5-4B` 下载底座（约 9.3 GB）。
若已有本地底座，把两个配置文件里的 `base_model_name_or_path` / `base_model`
改成你的本地路径即可。

> ⚠️ 本地重载必须显式传 `max_seq_length=4500`。Unsloth 的 `_load_clef` 会用
> `max_seq_length or CLEF_MAX_LEN` **无条件覆盖**保存的 `max_len`，不传就被降回 4096。
> 推理走 16384 不受影响，但用 `build_dataset` / `evaluate` 复现评测时会截断
> test 78 / ood 75 条。`predict.py` 已默认 4500。

期望输出（内置示例：客服工单，3 个问题一次直出）：

```
[choice] route        billing 0.973 ...
[score]  urgency      档位 2 ...
[noul]   refund_requested   P(为真) = 0.998 ...
```

---

## 复现训练

数据 parquet 不在本仓库分发（上游为来源依赖型混合许可，见下方「许可」）。
请从 Open-Jev 官方获取原始数据，再本地生成：

```bash
# 1) 原始 JSONL → typed 格式（需要上游 reproduce/source-code）
export OPEN_JEV_SRC=/path/to/Open-Jev/reproduce/source-code
export OPEN_JEV_RAW=/path/to/Open-Jev/raw
python scripts/to_typed_decisions.py --config release-v2-redistributable --parquet

# 2) 逐条裁决 + 分层抽样，生成 data/hybrid_v3/
python scripts/make_hybrid_v3.py

# 3) 校验零跳过、无跨集泄漏
python scripts/verify_v3.py

# 4) 训练（A10 22 GB，约 80 分钟含评测）
python -u scripts/train_decision.py \
  --data data/hybrid_v3 --model /path/to/Qwen3.5-4B --out runs/v3 \
  --steps 400 --max-seq 4500 --bs 2 --accum 8 --lr 2e-4 \
  --eval-every 100 --save-every 100 --eval-rows 0
```

云端（魔搭 DSW）一键复现见 [`docs/RETRAIN_GUIDE_V3.md`](docs/RETRAIN_GUIDE_V3.md)。

效率基准：

```bash
bash scripts/run_bench_v3.sh   # 需要把脚本内路径改成你的实际路径
```

---

## 目录

```
├── README.md                    本文件
├── docs/
│   ├── 微调全流程手册.md        完整流程、原理实测、排错、口径（附录 F = V3 记录）
│   └── RETRAIN_GUIDE_V3.md      云端复现指南
├── scripts/                     数据生成 / 审查 / 训练 / 评测 / 推理 / 基准
├── results/                     训练日志、曲线、分语料表、效率基准 JSON
├── data/hybrid_v3/manifest.json 五切分构成与抽样参数（parquet 需自行生成）
└── model/adapter_v3/            训练好的 LoRA + 决策头 + 温度（LFS）
```

---

## 口径与已知局限

写在这里是为了不误导，请照实引用：

- **只训了 0.41 epoch**（6,400 / 15,580 case），没有早停，曲线到 step 400 仍单调上升。
- 评测集**排除 painting**（4500 装不下 20k–60k token）：test 少 30.1% 题、ood 少 44.5% 题。
- 与官方 Open-Jev-2B / 9B 的公开数字**口径不同、不可比**：官方是全拆版、含 painting、
  10,046 行 hard reference matches；本仓库是原生多问题整条、排除 painting、7,136 题逐题 accuracy。
- `ood 0.9220 > test 0.8960` **不是泛化更强**，是评测集难度差异：test 有 472 个
  `tic_tac_toe`（需极小极大推演）而 ood 没有，且 ood 的 noul 占比更高。
- **没有做联合前向 vs 独立编码的消融，也没有拒识 / risk-coverage 评测**。
  「支持置信度阈值」仅指输出经过校准的概率，不代表做了阈值策略验证。
- 分语料表末尾「按题型分组」是逐题拆开后的诊断视图，**不是一次直出口径**。
- 温度只在独立 calibration 集按 3 个题型拟合并冻结，test/ood 各评一次。

---

## 许可

- 本仓库的**原创代码与文档**：MIT，见 [LICENSE](LICENSE)。
- 数据集：派生自 [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev)，
  上游为来源依赖型许可（CC0 / CC BY 4.0 / MIT 混合）。本仓库**不再分发 parquet**，
  只提供生成脚本与构成说明；使用时请遵循上游各来源的署名与许可要求。
- 底座：`Qwen/Qwen3.5-4B`，遵循 Qwen 许可。
- 模型权重：LoRA 适配器和决策头随仓库分发，同样受上述底座与数据许可约束。
- 框架：Unsloth / transformers / trl / bitsandbytes 各自许可。

本项目是**公开数据集上的方法验证**，不含真实业务数据，不代表任何生产部署。
