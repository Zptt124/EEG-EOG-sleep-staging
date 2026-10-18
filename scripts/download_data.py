"""Download Sleep-EDF Expanded sleep-cassette (SC) files from PhysioNet's
public S3 bucket (physionet-open) — the S3 endpoint is ~600x faster than
physionet.org for connections from China.

Downloads all 153 PSG + 153 hypnogram EDF files listed in SHA256SUMS.txt,
with resume support and sha256 verification. ~7.5 GB total.
"""
import concurrent.futures as cf
import hashlib
import os
import sys
import time
import urllib.request

from common import RAW

S3_BASE = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0"
RAW_DIR = str(RAW)
N_WORKERS = 16


def load_targets():
    """Parse SHA256SUMS.txt, keep sleep-cassette entries."""
    targets = []
    with open(os.path.join(RAW_DIR, "SHA256SUMS.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "sleep-cassette" not in line:
                continue
            sha, path = line.split()
            fname = path.split("/")[-1]
            targets.append((fname, sha.lower()))
    return targets


def fetch_one(fname, sha, retries=5):
    url = f"{S3_BASE}/sleep-cassette/{fname}"
    dest = os.path.join(RAW_DIR, fname)
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if os.path.exists(dest):
                # verify existing complete file
                if sha256_file(dest) == sha:
                    return (fname, "ok-cached", 0)
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if sha256_file(tmp) == sha:
                os.replace(tmp, dest)
                return (fname, "ok", os.path.getsize(dest))
            else:
                os.remove(tmp)
                return (fname, "bad-checksum", 0)
        except Exception as e:
            if attempt == retries - 1:
                return (fname, f"error:{type(e).__name__}", 0)
            time.sleep(2 * (attempt + 1))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    targets = load_targets()
    print(f"{len(targets)} files to download")
    done = fail = 0
    bytes_dl = 0
    t0 = time.time()
    # hypnograms first (tiny, needed for manifest/preprocess), then PSG
    targets.sort(key=lambda t: (not t[0].endswith("Hypnogram.edf"), t[0]))
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        futs = {ex.submit(fetch_one, f, s): f for f, s in targets}
        for fut in cf.as_completed(futs):
            fname, status, size = fut.result()
            bytes_dl += size
            done += 1
            if status.startswith("ok"):
                fail_mark = ""
            else:
                fail += 1
                fail_mark = "  <-- FAIL"
            elapsed = time.time() - t0
            rate = bytes_dl / elapsed / 1e6 if elapsed else 0
            print(f"[{done}/{len(targets)}] {fname} {status} "
                  f"({rate:.1f} MB/s){fail_mark}", flush=True)
    print(f"\nDONE: {done - fail} ok, {fail} failed")


if __name__ == "__main__":
    sys.exit(main())
