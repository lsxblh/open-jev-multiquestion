"""v3 数据验证：泄漏 / 长度 / build_dataset skipped。"""
import collections
import hashlib
import json
import os
import random

import pyarrow.parquet as pq
from unsloth import FastDecisionModel
from transformers import AutoTokenizer
from unsloth.models.clef import encode_record
from datasets import Dataset

ROOT = r"data/hybrid_v3"
SPLITS = ("train","calibration","validation","test","ood")

missing = [sp for sp in SPLITS
           if not os.path.exists(os.path.join(ROOT, sp+".parquet"))]
assert not missing, (
    f"缺少切分文件 {missing}；parquet 未在仓库分发，"
    "请按 README 用 make_hybrid_v3.py 生成后再运行本校验")

all_groups, all_states, all_samples = {}, {}, {}
for sp in SPLITS:
    rows = pq.read_table(os.path.join(ROOT, sp+".parquet")).to_pylist()
    all_groups[sp] = {r["group_id"] for r in rows}
    all_states[sp] = {hashlib.sha256(r["state"].encode("utf-8")).hexdigest()
                      for r in rows}
    all_samples[sp] = {
        hashlib.sha256((r["state"] + "||" + r["questions"]).encode("utf-8")).hexdigest()
        for r in rows}

pairs = [(a, b) for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]]
for label, index in (("group_id", all_groups),
                     ("state 文本", all_states),
                     ("(state, questions) 整条", all_samples)):
    leaks = [(a, b, len(index[a] & index[b])) for a, b in pairs if index[a] & index[b]]
    assert not leaks, f"{label} 跨 split 泄漏: {leaks}"
    print(f"[OK] {label}: {len(pairs)} 对切分两两交集全为 0", flush=True)

tok = AutoTokenizer.from_pretrained(r"model/adapter_v3")
for sp in SPLITS:
    rows = pq.read_table(os.path.join(ROOT, sp+".parquet")).to_pylist()
    random.seed(3407)
    sample = random.sample(rows, min(300, len(rows)))
    lens = [len(encode_record(tok, {"state":r["state"],
             "questions":json.loads(r["questions"]), "images":None},
             max_length=200000).input_ids) for r in sample]
    print(f"  {sp:12s} 抽样 {len(sample):3d} 条: max={max(lens):5d}, "
          f">4500 的 {sum(1 for x in lens if x>4500)} 条", flush=True)

model, proc = FastDecisionModel.from_pretrained(
    model_name=r"model/adapter_v3",
    max_seq_length=4500, load_in_4bit=True)
print("decision_config max_len =", model.decision_config.get("max_len"), flush=True)
for sp in SPLITS:
    rows = pq.read_table(os.path.join(ROOT, sp+".parquet")).to_pylist()
    sample = random.sample(rows, min(120, len(rows)))
    ds = Dataset.from_list([{k:r[k] for k in
        ("state","questions","gold","group_id","source","n_questions")} for r in sample])
    items, report = FastDecisionModel.build_dataset(ds, proc, model)
    print(f"  {sp:12s} items={len(items):3d}, skipped={report['skipped']}/{report['total']}", flush=True)
print("[DONE]", flush=True)
