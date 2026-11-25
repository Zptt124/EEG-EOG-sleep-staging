"""Step W4-B10: 无 EEG 组初步结果（B 在 Mac/MPS 上提的特征）与正式结果（D 在 N 卡上用 04 号提的特征）对比。

两边同一 30k EMA checkpoint、同一读出配方、同一 34 号探针和 head seed，只有特征提取的设备不同。
对每个预算比较：选中轮次 E*、α、开发集指标（原始与 α 调整后），以及逐段预测的一致率、p_N1 最大/平均绝对差。
输出：--out 下的 prelim_vs_formal.csv。

用法：
  python scripts/40_b_w4_prelim_vs_formal.py --prelim reports/week4/xmodal_prelim --formal runs --arm xm_none
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE

KEYS = ['macro_f1', 'n1_f1', 'n1_ap', 'n1_auroc']
ADJ = ['alpha', 'n1_f1_adjusted', 'macro_f1_adjusted']


def load(run_dir):
    m = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    p = pd.read_csv(run_dir / 'predictions_val.csv').sort_values('sample_id').reset_index(drop=True)
    return m, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', default=str(BASE / 'reports' / 'week4' / 'xmodal_prelim'))
    ap.add_argument('--formal', default=str(BASE / 'runs'))
    ap.add_argument('--arm', default='xm_none')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(BASE / 'reports' / 'week4' / 'xmodal_reeval'))
    args = ap.parse_args(argv)
    rows = []
    for budget in ('label10', 'label25'):
        rid = f'W4_B_{args.arm}_{budget}_sseed2024_seed{args.seed}_v01'
        mp, pp = load(Path(args.prelim) / rid)
        mf, pf = load(Path(args.formal) / rid)
        if not (pp['sample_id'].equals(pf['sample_id']) and pp['y_true'].equals(pf['y_true'])):
            raise SystemExit(f'{rid}: sample_id / y_true 不一致')
        row = dict(run_id=rid, budget=budget, epoch_prelim=mp['selected_epoch'], epoch_formal=mf['selected_epoch'])
        for k in KEYS:
            row.update({f'{k}_prelim': mp[k], f'{k}_formal': mf[k], f'd_{k}': mf[k] - mp[k]})
        for k in ADJ:
            row.update({f'{k}_prelim': mp['n1_adjust'][k], f'{k}_formal': mf['n1_adjust'][k],
                        f'd_{k}': mf['n1_adjust'][k] - mp['n1_adjust'][k]})
        dp = np.abs(pp['p_N1'].to_numpy() - pf['p_N1'].to_numpy())
        row.update(n_epochs=len(pp), y_pred_agree=float((pp['y_pred'] == pf['y_pred']).mean()),
                   p_n1_max_abs_diff=float(dp.max()), p_n1_mean_abs_diff=float(dp.mean()))
        rows.append(row)
        print(f"{budget}: E* {row['epoch_prelim']}→{row['epoch_formal']}  ΔMacro-F1 {row['d_macro_f1']:+.4f}  "
              f"ΔN1 AP {row['d_n1_ap']:+.4f}  α {row['alpha_prelim']:.2f}→{row['alpha_formal']:.2f}  "
              f"y_pred 一致 {row['y_pred_agree']:.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'prelim_vs_formal.csv', index=False, float_format='%.6f', lineterminator='\n')


if __name__ == '__main__':
    main()
