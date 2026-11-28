"""D：逐夜扫描 Ying 原始档案，寻找可与 EDF 时间轴对齐的关灯 / 上床事件，并填写 C 的事件登记模板。

对 20 个受试者压缩包里的每一晚（含第 3 晚，便于完整登记）：
  - 列出压缩包内全部文件（看有没有日志、事件、说明类文件）；
  - 读 EDF 头：起始日期与钟面时刻、受试者字段、记录字段、保留字段、信号数与各信号标签、数据记录数与时长；
  - 若有 "EDF Annotations" 信号，逐数据记录解析 TAL 文本，收集所有非空事件；
  - 读三位专家 TXT：首段起点、末段终点、出现过的所有标签文本（看是否夹带 Lights 等事件）；
  - 读三位专家 CSV：段数。
SLP010 第 1 晚另做对齐核对（EDF 时长、完整 30 秒段数、三位专家标注长度）。
输出：night_scan.csv、archive_files.csv、experiment3_time_events_D.csv（按 C 模板填写）、slp010_day1_alignment.json

用法：python 03_scan_time_events.py --raw <压缩包目录> --template <C 的模板 csv> --out <目录>
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

LIGHT_PAT = re.compile(r"light|lamp|bed|off|on\b|关灯|开灯|上床|起床", re.I)


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def edf_header(f):
    h = f.read(256)
    ns = int(h[252:256])
    sh = f.read(256 * ns)
    labels = [sh[i * 16:(i + 1) * 16].decode("latin-1").strip() for i in range(ns)]
    off = (16 + 80 + 8 * 5 + 80) * ns  # label, transducer, phys dim, phys min/max, dig min/max, prefilter
    nsamp = [int(sh[off + i * 8: off + (i + 1) * 8]) for i in range(ns)]
    return dict(patient=h[8:88].decode("latin-1").strip(), recording=h[88:168].decode("latin-1").strip(),
                date=h[168:176].decode(), time=h[176:184].decode(), header_bytes=int(h[184:192]),
                reserved=h[192:236].decode("latin-1").strip(), n_records=int(h[236:244]),
                rec_dur=float(h[244:252]), ns=ns, labels=labels, nsamp=nsamp)


def annotation_texts(f, hdr):
    """读取 EDF+ 注释信号中的全部 TAL 文本（假设头之后紧接数据记录）。"""
    if "EDF Annotations" not in hdr["labels"]:
        return None
    ai = hdr["labels"].index("EDF Annotations")
    rec_bytes = sum(n * 2 for n in hdr["nsamp"])
    a_off = sum(n * 2 for n in hdr["nsamp"][:ai])
    a_len = hdr["nsamp"][ai] * 2
    texts = []
    for _ in range(hdr["n_records"]):
        rec = f.read(rec_bytes)
        if len(rec) < rec_bytes:
            break
        tal = rec[a_off:a_off + a_len].split(b"\x00")
        for t in tal:
            parts = t.split(b"\x14")
            for p in parts[1:]:
                s = p.decode("latin-1").strip()
                if s:
                    texts.append(s)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    zips = sorted(a.raw.glob("SLP0*.zip"))
    if len(zips) != 20:
        fail(f"expected 20 archives, got {len(zips)}")
    nights, files_rows = [], []
    align = {}
    for zp in zips:
        z = zipfile.ZipFile(zp)
        names = z.namelist()
        for n in names:
            if not n.endswith("/"):
                files_rows.append(dict(archive=zp.name, file=n, size=z.getinfo(n).file_size))
        edfs = sorted(n for n in names if n.lower().endswith(".edf"))
        for e in edfs:
            night = Path(e).stem
            with z.open(e) as f:
                hdr = edf_header(f)
            with z.open(e) as f:
                f.read(hdr["header_bytes"])
                ann = annotation_texts(f, hdr)
            dur = hdr["n_records"] * hdr["rec_dur"]
            row = dict(night=night, edf_date=hdr["date"], edf_time=hdr["time"], duration_s=dur,
                       complete_epochs=int(dur // 30), patient_field=hdr["patient"], recording_field=hdr["recording"],
                       reserved_field=hdr["reserved"], n_signals=hdr["ns"], signal_labels="|".join(hdr["labels"]),
                       edf_annotation_channel=int(ann is not None), edf_annotation_events=len(ann or []),
                       edf_annotation_texts="|".join(sorted(set(ann or [])))[:300])
            for k in (1, 2, 3):
                base = e[:-4] + f"_Expert{k}"
                txt = z.read(base + ".txt").decode("utf-8", "replace").splitlines() if base + ".txt" in names else []
                csvl = z.read(base + ".csv").decode().splitlines() if base + ".csv" in names else []
                parts = [l.split("\t") for l in txt if l.strip()]
                row[f"expert{k}_txt_first_start"] = parts[0][1] if parts else ""
                row[f"expert{k}_txt_last_end"] = parts[-1][2] if parts else ""
                row[f"expert{k}_txt_labels"] = "|".join(sorted({p[0] for p in parts}))
                row[f"expert{k}_csv_epochs"] = len(csvl)
            labels_all = row["expert1_txt_labels"] + "|" + row["expert2_txt_labels"] + "|" + row["expert3_txt_labels"] + "|" + row["edf_annotation_texts"] + "|" + row["patient_field"] + "|" + row["recording_field"] + "|" + row["reserved_field"]
            other = sorted({t for t in labels_all.split("|") if t and t not in {"Wake", "NREM1", "NREM2", "NREM3", "REM", "X", "Startdate", "EDF+C", "EDF+D"}})
            row["non_stage_strings"] = "|".join(other)[:300]
            row["light_or_bed_like_strings"] = "|".join(t for t in other if LIGHT_PAT.search(t))[:300]
            nights.append(row)
            if night == "SLP010_Day1":
                align = dict(night=night, edf_start=f"{hdr['date']} {hdr['time']}", duration_s=dur,
                             complete_30s_epochs=int(dur // 30),
                             expert_csv_epochs=[row[f"expert{k}_csv_epochs"] for k in (1, 2, 3)],
                             expert_txt_first_start=[row[f"expert{k}_txt_first_start"] for k in (1, 2, 3)],
                             expert_txt_last_end=[row[f"expert{k}_txt_last_end"] for k in (1, 2, 3)])
        print(f"{zp.name}: {len(edfs)} nights scanned", flush=True)
    for name, rows in (("night_scan.csv", nights), ("archive_files.csv", files_rows)):
        with open(a.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (a.out / "slp010_day1_alignment.json").write_text(json.dumps(align, indent=2) + "\n")
    # 按 C 模板填写（只有第 1、2 晚）
    by = {r["night"]: r for r in nights}
    tpl = list(csv.DictReader(open(a.template, newline="", encoding="utf-8")))
    out = []
    for t in tpl:
        r = by.get(t["record_id"])
        if r is None:
            fail(f"template night {t['record_id']} not found in archives")
        found = r["light_or_bed_like_strings"]
        t = dict(t)
        if found:
            t.update(event_type="candidate_text_in_files", source_reference=f"{t['record_id']} files: {found}", status="needs_review")
        else:
            t.update(event_type="none_found",
                     source_reference=("EDF header start (date/time); no EDF Annotations events; expert TXT labels only sleep stages; "
                                       "archive contains only EDF/TXT/CSV; paper (PMC12141647) states lab schedule 23:30-08:30 and bedtime logs not released"),
                     status="no_per_night_event_available")
        t["edf_header_matches_template"] = int(r["edf_date"] == t["edf_start_date"] and r["edf_time"].replace(".", ":") == t["edf_start_local_clock"]
                                               and abs(float(r["duration_s"]) - float(t["recording_duration_seconds"])) < 1e-6)
        out.append(t)
    with open(a.out / "experiment3_time_events_D.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(json.dumps(dict(nights=len(nights), with_annotation_channel=sum(r["edf_annotation_channel"] for r in nights),
                          with_annotation_events=sum(r["edf_annotation_events"] > 0 for r in nights),
                          with_light_like_strings=sum(bool(r["light_or_bed_like_strings"]) for r in nights),
                          non_media_files=sorted({Path(r["file"]).suffix for r in files_rows}),
                          template_header_matches=sum(t["edf_header_matches_template"] for t in out), template_rows=len(out),
                          slp010_day1=align), ensure_ascii=False))


if __name__ == "__main__":
    main()
