"""B 第 3 周：独立下载并校验 Ying OSF 原始包（gx36m）。

与 C 的 download.py 无共享代码。下载 OSF 根目录全部文件（20 个 SLP*.zip 与说明、
专家信息表等），逐个用 OSF 元数据里的 SHA-256 与字节数校验，并把 OSF 记录的创建/修改
时间写进清单，作为协议时间线的外部证据。

用法：python -I scripts/54_b_w3_ying_download.py cache/ying_osf
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.osf.io/v2/nodes/gx36m/files/osfstorage/?page%5Bsize%5D=100"
UA = {"User-Agent": "sleep-staging-replication/1.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def listing():
    files, url = [], API
    while url:
        j = get_json(url)
        files += [f for f in j["data"] if f["attributes"]["kind"] == "file"]
        url = j["links"]["next"]
    return files


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def fetch(f, raw):
    a = f["attributes"]
    dest = raw / a["name"]
    want = a["extra"]["hashes"]["sha256"]
    if dest.exists() and dest.stat().st_size == a["size"] and sha256(dest) == want:
        return a["name"], "cached"
    tmp = dest.with_name(dest.name + ".partial")
    for attempt in range(5):
        try:
            req = urllib.request.Request(f["links"]["download"], headers=UA)
            h = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                while b := r.read(1 << 20):
                    w.write(b)
                    h.update(b)
            if tmp.stat().st_size != a["size"] or h.hexdigest() != want:
                raise ValueError("checksum mismatch")
            tmp.replace(dest)
            return a["name"], "downloaded"
        except Exception as e:  # noqa: BLE001
            print(f"{a['name']} attempt {attempt + 1}: {type(e).__name__} {str(e)[:120]}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(a["name"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    raw = args.directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = listing()
    names = sorted(f["attributes"]["name"] for f in files)
    slp = [n for n in names if n.startswith("SLP") and n.endswith(".zip")]
    if slp != [f"SLP{i:03d}.zip" for i in range(1, 21)]:
        sys.exit(f"Unexpected SLP archive set: {slp}")
    t0 = time.monotonic()
    total = sum(f["attributes"]["size"] for f in files)
    print(f"{len(files)} files, {total / 2**30:.2f} GiB", flush=True)
    with ThreadPoolExecutor(6) as ex:
        for name, status in ex.map(lambda f: fetch(f, raw), files):
            print(f"{name} {status} {time.monotonic() - t0:.0f}s", flush=True)
    manifest = [
        {
            "name": f["attributes"]["name"],
            "size": f["attributes"]["size"],
            "sha256": f["attributes"]["extra"]["hashes"]["sha256"],
            "md5": f["attributes"]["extra"]["hashes"].get("md5"),
            "osf_date_created": f["attributes"]["date_created"],
            "osf_date_modified": f["attributes"]["date_modified"],
            "osf_version": f["attributes"].get("current_version"),
            "local_sha256_verified": sha256(raw / f["attributes"]["name"]) == f["attributes"]["extra"]["hashes"]["sha256"],
        }
        for f in sorted(files, key=lambda f: f["attributes"]["name"])
    ]
    if not all(m["local_sha256_verified"] for m in manifest):
        sys.exit("Verification failed")
    (args.directory / "osf_manifest.json").write_text(
        json.dumps({"node": "gx36m", "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "files": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("All files verified", flush=True)


if __name__ == "__main__":
    main()
