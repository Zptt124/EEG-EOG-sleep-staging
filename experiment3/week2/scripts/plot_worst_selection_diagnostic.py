"""Plot the two already completed validation histories; never rerun or reselect."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/week2_final_C_20261029'


def main():
    finding=json.loads((OUT/'key_findings.json').read_text())['worst_primary_N1_pair']
    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    for key,label,color in [('S0_run_id','S0','#39739f'),('S1_run_id','S1, adaptation seed 0','#b65d49')]:
        folder=ROOT/'runs'/finding[key]
        history=json.loads((folder/'history.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        x=[h['epoch'] for h in history];y=[100*h['val_macro_f1'] for h in history]
        best=summary['best_epoch']
        ax.plot(x,y,label=f'{label}; selected epoch {best}',color=color,linewidth=1.7)
        ax.scatter([best],[y[x.index(best)]],s=45,color=color,zorder=5)
        ax.axvline(best,color=color,alpha=.3,linestyle='--')
    ax.set(title='Primary 5-subject budget, head seed 0: recorded model-selection histories',
           xlabel='Completed classifier epoch',ylabel='Development Macro-F1 (%)')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
    fig.savefig(OUT/'worst_pair_validation_selection.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
