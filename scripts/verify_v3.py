"""v3 数据验证：泄漏 / 长度 / build_dataset skipped。"""
import collections
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
all_groups = {}
for sp in SPLITS:
    rows = pq.read_table(os.path.join(ROOT, sp+".parquet")).to_pylist()
    all_groups[sp] = {r["group_id"] for r in rows}
leaks = [(a,b,len(all_groups[a]&all_groups[b]))
         for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]
         if all_groups[a]&all_groups[b]]
print("跨 split group 泄漏:", "无" if not leaks else leaks, flush=True)

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
