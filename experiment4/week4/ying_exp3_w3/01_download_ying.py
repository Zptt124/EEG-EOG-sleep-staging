"""D：从 OSF 官方档案下载 Ying 20 个受试者压缩包，按 C 交接的 osf_files.json 逐文件核对大小和 SHA-256。
已验证的文件跳过；失败重试 4 次后 sys.exit(1)。

用法：python 01_download_ying.py --files-json <osf_files.json> --out <下载目录>
"""
import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [f for f in json.loads(a.files_json.read_text()) if f["attributes"]["name"].startswith("SLP")]
    if len(files) != 20:
        print(f"ERROR: expected 20 subject archives, got {len(files)}", file=sys.stderr)
        sys.exit(1)

    def get(f):
        at = f["attributes"]
        dest = a.out / at["name"]
        want = at["extra"]["hashes"]["sha256"]
        if dest.exists() and dest.stat().st_size == at["size"] and sha(dest) == want:
            return f"{at['name']} already verified"
        tmp = dest.with_suffix(".partial")
        for i in range(4):
            try:
                req = Request(f["links"]["download"], headers={"User-Agent": "Mozilla/5.0 (research client)"})
                with urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                    shutil.copyfileobj(r, w, length=1 << 20)
                if tmp.stat().st_size != at["size"] or sha(tmp) != want:
                    raise ValueError("size or checksum mismatch")
                tmp.replace(dest)
                return f"{at['name']} downloaded and verified"
            except Exception as e:
                print(f"{at['name']} attempt {i + 1} failed: {type(e).__name__} {str(e)[:120]}", flush=True)
        return f"FAILED {at['name']}"

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(get, sorted(files, key=lambda f: f["attributes"]["name"])))
    for r in res:
        print(r, flush=True)
    if any(r.startswith("FAILED") for r in res):
        print("ERROR: some archives failed", file=sys.stderr)
        sys.exit(1)
    print("ALL 20 ARCHIVES VERIFIED", flush=True)


if __name__ == "__main__":
    main()
