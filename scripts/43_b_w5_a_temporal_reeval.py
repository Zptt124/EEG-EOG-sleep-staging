"""Step W5-B: 统一复评 A 第五周 temporal context 三臂（PR #28，single / cat ±2 / BiGRU）。

复用 39 号的逐 run 核对与配对流程（官方 evaluate.py v1 对照 val_reference、y_pred=argmax、预测 SHA、
inner_cv 重选 E*、五折划分、OOF 受试者、28 号网格独立重选 α 与调整后指标、按 8 名开发受试者成对 bootstrap
1000 次、rng seed 0），只把三组 arm 与三个对比换成 A 的设置：
  bgru−single、bgru−cat、cat−single（原始与 α 调整后各一遍）。
A 的登记表用 Windows 路径分隔符，这里先转成 '/' 再交给 39 号。

用法（--root 指向 A 分支导出的 reports/week5/a_temporal_ctx）：
  python scripts/43_b_w5_a_temporal_reeval.py --root <A 分支>/reports/week5/a_temporal_ctx \
      --ref-paired eval/paired.csv --out reports/week5/b_a_tctx_reeval
"""
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

ARMS = {'bgru': 'tctx_bgru', 'cat': 'tctx_cat', 'single': 'tctx_single'}
CONTRASTS = [('bgru', 'single'), ('bgru', 'cat'), ('cat', 'single')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--registry', default='eval/registry.csv')
    ap.add_argument('--ref-paired', default='eval/paired.csv')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location('reeval39', Path(__file__).with_name('39_b_w4_xmodal_reeval.py'))
    m39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m39)
    m39.ARMS, m39.CONTRASTS = ARMS, CONTRASTS

    reg = pd.read_csv(Path(args.root) / args.registry)
    reg['predictions'] = reg['predictions'].str.replace('\\', '/', regex=False)
    with tempfile.TemporaryDirectory() as td:
        reg_path = Path(td) / 'registry.csv'
        reg.to_csv(reg_path, index=False, lineterminator='\n')
        m39.main(['--root', args.root, '--registry', str(reg_path.resolve()),
                  '--ref-paired', args.ref_paired, '--out', args.out])

    # 39 号把脚本名写成自己，这里改回 43 号并记下登记表来源
    run_json = Path(args.out) / 'reeval_run.json'
    txt = run_json.read_text(encoding='utf-8')
    txt = txt.replace('scripts/39_b_w4_xmodal_reeval.py', 'scripts/43_b_w5_a_temporal_reeval.py（复用 39 号流程）')
    txt = txt.replace(str(reg_path.resolve()), f'{args.registry}（路径分隔符转为 /）')
    run_json.write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
