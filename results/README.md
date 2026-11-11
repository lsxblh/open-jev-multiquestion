# 结果文件说明

## V3（当前版本，2026-11-06 训练）

- `v3_run_report.json` — 最终 test / ood / calibration 指标、训练耗时
- `v3_run_config.json` — 训练超参
- `v3_metrics_log.jsonl` — step 100/200/300/400 的 validation 曲线
- `v3_train.log` — 云端完整训练日志
- `v3_breakdown_test.txt` / `v3_breakdown_ood.txt` — 分语料 / 分题型评测
- `v3_inventory.md` — 五切分构成盘点
- `v3_latency_bench.json` — 本地 RTX 3060 6GB 推理基准
- `v3_latency_a10.json` — A10 短请求 + 批量吞吐
- `v3_latency_long_a10.json` — A10 真实长序列三档（p50/p90/max）
- `v3_cold_a10.json` — A10 冷启动
- `split_legality.md` — 逐条审读 `instructions` 的拆分合法性证据
- `data_samples.md` — 8 个 source 的真实样本原文

## v2 历史（全拆版口径，已废弃，**不可与 V3 比较**）

`run_report.json`、`metrics_log.jsonl`、`breakdown_test.txt`、
`latency_bench.json`、`cold_latency.json`。

这些是 2026-10-30 那轮「每行 1 题、`max_seq=1024`、含 painting」的结果，
保留仅供原理走查与历史对照。

## 原理实测（与数据版本无关，长期有效）

`attention_trace.json`、`fewshot_trace.json`、`tensor_shapes.json`、
`joint_vs_split.json`、`batch_decision_detail.json`、
`batch_decision_truth.json`、`per_question_batch.json`、
`one_case_four_shapes.md`。

对应手册附录 B–E 的张量走查与效率基准。
