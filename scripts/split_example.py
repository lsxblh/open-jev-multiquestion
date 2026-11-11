"""实例：一个 case 拆分前 / 拆分后的真实形态。"""
import json

import pyarrow.parquet as pq

FULL = r"data/typed_full\train.parquet"
SPLIT = r"data/typed_split\train.parquet"
full = pq.read_table(FULL).to_pylist()
split = pq.read_table(SPLIT).to_pylist()

print("typed_full train:", len(full), "cases")
print("typed_split train:", len(split), "cases")

# 例 A：workflow-controls，找问题数最多的 case
best = max((r for r in full if r["source"].startswith("workflow-controls")),
           key=lambda r: len(json.loads(r["questions"])))
print("\n" + "=" * 76)
print("例 A：拆分前（一个 case 里塞了多个问题）")
print("=" * 76)
print("source :", best["source"])
print("group_id :", best["group_id"])
print("n_questions :", best["n_questions"])
print("state (前 500 字):")
print("  ", best["state"][:500])
qs = json.loads(best["questions"])
golds = json.loads(best["gold"])
print("\nquestions（", len(qs), "个）:")
for name, q in qs.items():
    crit = q.get("criteria")
    n = len(crit) if isinstance(crit, (dict, list)) else 2
    print(f"  {name:28s} type={q['type']:6s} 候选数={n}  "
          f"label={golds[name].get('label')}")
print("\ngold 里第一项的完整结构:")
k0 = list(golds)[0]
print(" ", k0, "->", json.dumps(golds[k0], ensure_ascii=False)[:220])

print("\n" + "=" * 76)
print("例 A：拆分后（同一个 group_id 变成多少个独立 case）")
print("=" * 76)
children = [r for r in split if r["group_id"] == best["group_id"]]
print("拆出", len(children), "个 case，group_id 完全相同 =", {c["group_id"] for c in children})
for i, c in enumerate(children[:5]):
    q = json.loads(c["questions"])
    name = list(q)[0]
    g = json.loads(c["gold"])
    print(f"  [{i}] questions 只剩 1 个: {name:24s} type={q[name]['type']:6s} "
          f"label={g[name].get('label')}")
    print(f"      state 与父 case 完全相同? {c['state'] == best['state']}")
    print(f"      n_questions = {c['n_questions']}")
