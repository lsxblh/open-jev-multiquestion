"""全库实测：不拆分的话，多少 case 能装进各种 max_seq_length。"""
import collections

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from unsloth.models.clef import encode_record

tok = AutoTokenizer.from_pretrained(r"model/adapter_v3")
rows = pq.read_table(r"data/typed_full\train.parquet").to_pylist()
print("typed_full train cases:", len(rows), flush=True)

import json
lens = []
by_src = collections.defaultdict(list)
for i, r in enumerate(rows):
    qs = json.loads(r["questions"])
    enc = encode_record(tok, {"state":r["state"],"questions":qs,"images":None}, max_length=200000)
    n = len(enc.input_ids)
    lens.append(n)
    by_src[r["source"]].append(n)
    if (i+1) % 4000 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

lens.sort()
def pct(p):
    return lens[int(len(lens)*p)-1]
print("\n全库编码长度: p50=%d p90=%d p99=%d max=%d" % (pct(.5), pct(.9), pct(.99), lens[-1]), flush=True)
print("\n各 max_seq_length 下能装下的 case 数:")
for lim in (1024, 2048, 4096, 8192, 16384):
    ok = sum(1 for x in lens if x <= lim)
    print(f"  {lim:6d}: {ok:6d}/{len(lens)}  ({100*ok/len(lens):5.1f}%)", flush=True)

print("\n各 source 的 p50 / max:")
for s, v in sorted(by_src.items()):
    v.sort()
    print(f"  {s:48s} n={len(v):5d}  p50={v[len(v)//2]:6d}  max={v[-1]:6d}", flush=True)
