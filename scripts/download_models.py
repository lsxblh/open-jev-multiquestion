"""从 ModelScope 下载指定的 Qwen3.5 底座。"""
import json
import os
import sys
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}
REPO = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Qwen3.5-4B"
DEST = sys.argv[2] if len(sys.argv) > 2 else r"models/Qwen3.5-4B"
os.makedirs(DEST, exist_ok=True)

url = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Root="
req = urllib.request.Request(url, headers=HEADERS)
with urllib.request.urlopen(req, timeout=60) as r:
    files = json.loads(r.read().decode("utf-8"))["Data"]["Files"]

for f in files:
    if f["Type"] != "blob" or f["Name"].startswith("."):
        continue
    name = f["Path"]
    size = f["Size"]
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) == size:
        print("skip", name, flush=True)
        continue
    print(f"downloading {name}  {size/1e6:.1f} MB", flush=True)
    rq = urllib.request.Request(
        f"https://modelscope.cn/models/{REPO}/resolve/master/{name}", headers=HEADERS)
    with urllib.request.urlopen(rq, timeout=600) as resp, open(out, "wb") as fh:
        fh.write(resp.read())
    print("  done", os.path.getsize(out), flush=True)
print("ALL DONE ->", DEST, flush=True)
