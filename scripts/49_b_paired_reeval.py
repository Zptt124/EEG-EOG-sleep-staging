"""通用复评入口：对任意两组或多组 34 号协议的 run 复用 39 号逐 run 核对与配对流程（43 号的通用版）。

arm 与对比从命令行给出；登记表只保留这些 arm 的行，Windows 路径分隔符转为 '/'。
39 号的核对见其说明（官方 evaluate.py v1、y_pred=argmax、预测 SHA、重选 E*、五折、OOF 受试者、28 号重选 α、
调整后指标、按 8 名开发受试者成对 bootstrap 1000 次、rng seed 0）。

用法（A 的 W6 timestep 筛查，PR #29）：
  python scripts/49_b_paired_reeval.py --root <A 分支>/reports/week6/a_tsampling \
      --arms tlogit_all=w6_tlogit_all base_all=w6_base_all --contrasts tlogit_all-base_all \
      --ref-paired eval/paired_all.csv --out reports/week6/b_a_tsampling_reeval/all
"""
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--registry', default='eval/registry.csv')
    ap.add_argument('--arms', nargs='+', required=True, help='短名=登记表 arm 名')
    ap.add_argument('--contrasts', nargs='+', required=True, help='短名-短名')
    ap.add_argument('--ref-paired')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    arms = dict(a.split('=', 1) for a in args.arms)
    contrasts = [tuple(c.split('-', 1)) for c in args.contrasts]
    if any(a not in arms or b not in arms for a, b in contrasts):
        sys.exit('对比里出现未定义的 arm 短名')

    spec = importlib.util.spec_from_file_location('reeval39', Path(__file__).with_name('39_b_w4_xmodal_reeval.py'))
    m39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m39)
    m39.ARMS, m39.CONTRASTS = arms, contrasts

    reg = pd.read_csv(Path(args.root) / args.registry)
    reg = reg[reg['arm'].isin(arms.values())].copy()
    reg['predictions'] = reg['predictions'].str.replace('\\', '/', regex=False)
    with tempfile.TemporaryDirectory() as td:
        reg_path = Path(td) / 'registry.csv'
        reg.to_csv(reg_path, index=False, lineterminator='\n')
        argv = ['--root', args.root, '--registry', str(reg_path.resolve()), '--out', args.out]
        if args.ref_paired:
            argv += ['--ref-paired', args.ref_paired]
        m39.main(argv)
    run_json = Path(args.out) / 'reeval_run.json'
    txt = run_json.read_text(encoding='utf-8')
    txt = txt.replace('scripts/39_b_w4_xmodal_reeval.py', 'scripts/49_b_paired_reeval.py（复用 39 号流程）')
    txt = txt.replace(str(reg_path.resolve()), f'{args.registry}（按 arm 过滤，路径分隔符转为 /）')
    run_json.write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
