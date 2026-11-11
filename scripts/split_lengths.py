"""实测拆分前/后的 encode_record 长度。"""
import json

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from unsloth.models.clef import encode_record

tok = AutoTokenizer.from_pretrained(r"model/adapter_v3")
full = pq.read_table(r"data/typed_full\train.parquet").to_pylist()

def ntokens(state, questions):
    rec = {"state": state, "questions": questions, "images": None}
    enc = encode_record(tok, rec, max_length=65536)
    return len(enc.input_ids), len(enc.questions), sum(len(q.option_spans) for q in enc.questions)

# 例 A：workflow-controls 17 个问题
p = max((r for r in full if r["source"].startswith("workflow-controls")),
        key=lambda r: len(json.loads(r["questions"])))
qs = json.loads(p["questions"])
n_all, nq, nopt = ntokens(p["state"], qs)
print("=== 例 A: workflow-controls ===")
print(f"  拆分前: {nq} 个问题 / {nopt} 个候选 -> {n_all} token")
one = {list(qs)[0]: qs[list(qs)[0]]}
n_one, nq1, nopt1 = ntokens(p["state"], one)
print(f"  拆分后: 1 个问题 / {nopt1} 个候选 -> {n_one} token   (每个 case)")
print(f"  17 个 case 合计 ≈ {n_one*17} token（state 被重复编码 17 次）")

# 例 B：painting-geometry（超长元凶）
pg = [r for r in full if r["source"] == "painting-geometry-v1"]
print(f"\n=== 例 B: painting-geometry ({len(pg)} 个 case) ===")
big = max(pg, key=lambda r: len(json.loads(r["questions"])))
bqs = json.loads(big["questions"])
nb, nqb, noptb = ntokens(big["state"], bqs)
print(f"  最大的 case: {nqb} 个问题 / {noptb} 个候选 -> {nb} token")
one_name = list(bqs)[0]
nb1, _, noptb1 = ntokens(big["state"], {one_name: bqs[one_name]})
print(f"  拆成单问题后: 1 个问题 / {noptb1} 个候选 -> {nb1} token")
print(f"  压缩比: {nb}/{nb1} = {nb/nb1:.0f}x")
lens = []
for r in pg:
    q = json.loads(r["questions"])
    n, _, _ = ntokens(r["state"], q)
    lens.append(n)
lens.sort()
print(f"  全部 {len(pg)} 个 case 编码长度: p50={lens[len(lens)//2]}, "
      f"p90={lens[int(len(lens)*0.9)]}, max={lens[-1]}")
print(f"  max_seq_length=1024 时: 拆分前 {sum(1 for x in lens if x>1024)}/{len(lens)} 装不下")

# 拆分后全库最大长度（抽样）
split = pq.read_table(r"data/typed_split\train.parquet").to_pylist()
import random
random.seed(3407)
sample = random.sample(split, 400)
sl = []
for r in sample:
    n, _, _ = ntokens(r["state"], json.loads(r["questions"]))
    sl.append(n)
sl.sort()
print(f"\n=== 拆分后全库（抽样 400 条）===")
print(f"  p50={sl[len(sl)//2]}  p90={sl[int(len(sl)*0.9)]}  max={sl[-1]}")
print(f"  >1024 的条数: {sum(1 for x in sl if x>1024)}")
