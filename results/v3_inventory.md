# V3 数据集盘点

数据目录：`jev/data/hybrid_v3/`，`max_seq=4500`，规则见 `manifest.json`。

## 1. 五个切分

| split | case 数 | 题数 | N=1 case | N>1 case | 每题最多候选 | groups | 文件 KB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train` | 15,580 | 55,523 | 5,815 | 9,765 | 16 | 7,256 | 612 |
| `calibration` | 1,078 | 4,086 | 371 | 707 | 7 | 467 | 70 |
| `validation` | 932 | 3,145 | 344 | 588 | 7 | 451 | 66 |
| `test` | 1,902 | 7,136 | 678 | 1,224 | 7 | 887 | 105 |
| `ood` | 1,772 | 8,627 | 0 | 1,772 | 8 | 1,396 | 103 |

## 2. train 的构成：原生 case + painting 拆出的片段

- **原生 case：14,403**（1 state + 多问题，不拆不合并）
- **painting 单问题片段：1,177**（23,552 条里分层抽 5%，seed 3407）
- **合计：15,580**

| 原生 case 按 source | case 数 | 题数 | 题型 |
|---|---:|---:|---|
| `snake-v1` | 3,127 | 12,508 | noul 9,381/choice 3,127 |
| `workflow-controls-v1/security_incidents` | 522 | 8,874 | noul 8,874 |
| `vizdoom-basic-v1` | 2,118 | 6,354 | choice 2,118/noul 2,118/score 2,118 |
| `reasoning-control-v1` | 1,814 | 5,442 | choice 1,814/noul 1,814/score 1,814 |
| `workflow-controls-v1/invoice_processing` | 500 | 5,000 | noul 5,000 |
| `customer-control-v1` | 701 | 4,206 | score 2,103/noul 1,402/choice 701 |
| `workflow-controls-v1/agent_trace_observability` | 540 | 3,780 | noul 3,780 |
| `workflow-controls-v1/customer_service` | 443 | 3,544 | noul 3,544 |
| `tic_tac_toe-v1` | 3,264 | 3,264 | choice 3,264 |
| `tile_platformer-v1` | 1,346 | 1,346 | choice 1,346 |
| `trex_runner-v1` | 28 | 28 | choice 28 |

## 3. 全库题型分布（V3）

| split | choice | score | noul | 合计 |
|---|---:|---:|---:|---:|
| `train` | 12,692 | 6,329 | 36,502 | 55,523 |
| `calibration` | 925 | 502 | 2,659 | 4,086 |
| `validation` | 829 | 353 | 1,963 | 3,145 |
| `test` | 1,640 | 816 | 4,680 | 7,136 |
| `ood` | 1,491 | 825 | 6,311 | 8,627 |

## 4. 评测集相对原始切分排除了什么

排除的只有 painting（4500 装不下），其余原样保留。

| split | V3 case | 原始 case | 排除 case | 排除题数 | 占原始题数 |
|---|---:|---:|---:|---:|---:|
| `calibration` | 1,078 | 1,081 | 3 | 512 | 11.1% |
| `validation` | 932 | 935 | 3 | 512 | 14.0% |
| `test` | 1,902 | 1,920 | 18 | 3,072 | 30.1% |
| `ood` | 1,772 | 1,790 | 18 | 6,912 | 44.5% |

## 5. 一条 case 的字段

```text
state      情境文本（JSON 字符串或对话）
questions  {问题名: {type, instructions, criteria}}   JSON 字符串
gold       {问题名: {type, label, probabilities, ...}} JSON 字符串
group_id   同一原始 case 的分身共享，防跨 split 泄漏
source     12 选 1（评测集里 painting 已排除，剩 11 个）
n_questions
```

题型：`choice`（多项，criteria 是 dict）、`score`（有序档位，criteria 是 list）、`noul`（布尔，无 criteria）。
