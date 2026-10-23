"""Create final reproducibility records, retrospective, figures and small D supplement."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final_week1'
B=ROOT/'vendor/experiment2-review-6e7df99'
FOLLOW=ROOT/'reports/b_followup_20261023'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')

def dump(path,data):
    write(path,json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    adapters=load(OUT/'adapter_checks.json')
    assert adapters['status']=='passed'
    follow=load(FOLLOW/'summary.json')
    assert follow['status']=='complete' and follow['completed_sampling_seeds']==[2024,2025,2026]
    assert all(r['status']=='passed' for r in load(FOLLOW/'new_heads_replay_audit.json').values())
    cfg=load(ROOT/'features/eeg_cond14_v01/feature_config.json')
    text=yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)
    write(ROOT/'configs/feature_config.yaml',text)
    profile=pd.read_csv(ROOT/'features/eeg_cond14_v01/resource_profile.csv')
    assert len(profile)==122 and int(profile.n_samples.sum())==154124
    resources=dict(samples=154124,records=122,feature_bytes=int(profile.bytes.sum()),
        extraction_compute_and_write_seconds=float(profile.seconds.sum()),
        throughput_samples_per_second=float(profile.n_samples.sum()/profile.seconds.sum()),
        peak_gpu_mb=float(profile.peak_gpu_mb.max()),
        raw_float32_bytes_without_headers=154124*20*30*128*4,
        stored_float16_bytes_without_headers=154124*20*30*128*2,
        cost_of_mean_pooled_float32_bytes=154124*20*128*4)
    dump(OUT/'resource_summary.json',resources)
    hashes={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [
        ROOT/'scripts/extract_features.py',ROOT/'scripts/train_linear100.py',ROOT/'scripts/train_linear_budget.py',
        ROOT/'scripts/train_timeflatten.py',ROOT/'scripts/run_b_followup.py',ROOT/'scripts/complete_adapter_checks.py']}
    dump(OUT/'code_sha256.json',hashes)
    runs=[]
    base='E3_EEG_label100_seed0_provisional_v01'
    flat='E3_EEG_label100_seed0_timeflatten_v01'
    names=[base,flat]+[f'E3_EEG_label25_sseed{s}_seed0_v01' for s in (2024,2025,2026)]
    for name in names:
        run=ROOT/'runs'/name
        original_cfg=load(run/'run_config.json')
        conf=dict(original_cfg)
        conf.update(evaluation='B official v1',b_commit=follow['b_commit'],
            original_config='run_config.json (preserved historical record)',
            checkpoint_alias='best.ckpt is a byte-identical copy of best_head.pt')
        write(run/'config.yaml',yaml.safe_dump(conf,allow_unicode=True,sort_keys=False))
        history=load(run/'history.json')
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        shutil.copy2(run/'best_head.pt',run/'best.ckpt')
        metric_path=FOLLOW/('metrics_official_E3_100.json' if name==base else 'metrics_official_E3_timeflatten100.json') if name in (base,flat) else run/'metrics_official.json'
        metric=load(metric_path)
        dump(run/'metrics.json',metric)
        summary=load(run/'summary.json')
        assert summary['status']=='complete'
        pred=pd.read_csv(run/'predictions_val.csv')
        assert len(pred)==21186 and pred.sample_id.is_unique
        runnote=[f'# {name}','', '负责人：C（实验③）；状态：成功，验证集实验完成。',
            '', '原始计算日期：100%及时间展开为2026-10-20；25%为2026-10-23。记录与正式评估整理日期：2026-10-23。',
            '', f"标签预算：{int(original_cfg['label_fraction']*100)}%；训练人数：{len(original_cfg['training_subjects'])}；训练段数：{summary['train_samples']}；验证：8人、21186段。",
            '', f"训练seed=0；名单抽样种子={original_cfg.get('sampling_seed','100%不适用')}；B名单与划分版本见config.yaml。",
            '', f"最佳第{summary['best_epoch']}轮，Macro-F1={metric['macro_f1']:.10f}，N1-F1={metric['n1_f1']:.10f}。最佳权重选择依据为验证Macro-F1。",
            '', f"主干固定为官方EMA权重，配置与SHA见config.yaml；分类头在CPU训练。E3上游代码commit={cfg['upstream_commit']}，A数据commit={cfg['source_commit']}，B评估commit={follow['b_commit']}。本机胶水脚本未提交Git，文件哈希见reports/final_week1/code_sha256.json。",
            '', '文件：config.yaml、history.csv、best.ckpt、metrics.json、predictions_val.csv为团队约定格式；原run_config.json、history.json、best_head.pt及原summary.json保留。best.ckpt不是新训练权重，和best_head.pt字节一致。',
            '', 'Python3.11.17、PyTorch2.7.1+cu126；完整依赖见environment/requirements_freeze_wsl.txt。主干使用GPU，batch4，资源汇总见reports/final_week1/resource_summary.json；分类头计时见原summary.json。',
            '', '原始运行与复现：进入experiment3后执行下方命令，复现必须选新输出目录，避免覆盖已完成结果。']
        if name==base:
            cmd='python scripts/train_linear100.py --output runs/reproduce_E3_mean100'
        elif name==flat:
            cmd='python scripts/train_timeflatten.py'
            runnote += ['', '时间展开脚本的OUT与CACHE为固定路径：复现前需在脚本顶部将OUT改为新目录，CACHE在Linux中可复用。无需重提主干。']
        else:
            cmd=f"python scripts/train_linear_budget.py --sampling-seed {original_cfg['sampling_seed']} --output runs/reproduce_{name}"
        runnote += ['', '```bash',cmd,'```',
            '', '正式评估命令（把RUN目录替换为对应复现输出目录）：',
            '', '```bash','python vendor/experiment2-review-6e7df99/src/evaluation/evaluate.py runs/RUN/predictions_val.csv --out runs/RUN/metrics.json --bootstrap 1000 --seed 0','```',
            '', '每组均保留原始验证预测；未使用测试集。验证集用于选模，只有单训练种子；EEGDM和CNN不是等架构/等预训练数据的严格对照。']
        write(run/'README.md','\n'.join(runnote)+'\n')
        runs.append(dict(run_id=name,budget=original_cfg['label_fraction'],sampling_seed=original_cfg.get('sampling_seed'),
            train_subjects=len(original_cfg['training_subjects']),train_samples=summary['train_samples'],
            macro_f1=metric['macro_f1'],n1_f1=metric['n1_f1'],best_epoch=summary['best_epoch'],
            train_seconds=summary.get('training_seconds'),path='runs/'+name))
    pd.DataFrame(runs).to_csv(OUT/'run_index.csv',index=False)

    note='''# EEGDM输入与表征理解笔记

EEGDM在本周只作为冻结特征提取器，加载backbone.ckpt的EMA主干，严格匹配参数。没有在Sleep-EDF上重新训练或微调主干；仅训练五类线性分类器。

原始睡眠片段来自A的NPZ，signal为[N,2,3000]，100Hz、30秒、µV、未z-score。EEG取下标0，EOG下标1留给D。先用resample_poly上采样到200Hz，再除100，调用一次官方staged_mu_law。该函数对归一化后|x|≤1保留原值，不能用普通全域mu-law替代。

模型条件编号和数据通道下标不同。主方案use_cond=[14]，零基编号FP1-F3，作为Fpz-Cz的代理条件；候选18为FP2-F4，10为C3-CZ。没有Fpz-Cz专用预训练条件。小样本候选比较不等于证明真实通道条件最优。

输入一次传入6000点，但官方提取器内部拆成6×1000点，即6个5秒片段，输出按原时间顺序重组。因此“30秒调用”不具有连续30秒主干上下文；本周已经检验原生调用与显式六段调用的数值等价性。

取20层gate激活，零基0–19；每200点（1秒）计算一次标准差，步长200点，无重叠，torch correction=1。t=1是扩散步条件，但当前调用不向输入添加随机噪声。原任务文档将t=1概述为轻噪声，实际实现以核对过的代码为准，不能写成已经加噪。

每段完整特征[20,30,128]，float32计算、float16存储。全量按记录保存，feature_index.csv通过sample_id关联原始epoch_index和行号。主分类头先在时间维取均值，形成2560维，再使用单层Linear(2560,5)。特征标准化仅由对应有标签训练受试者拟合，验证样本不能参与拟合。

S4自定义CUDA扩展未安装，使用官方回退实现。冻结S4卷积核缓存只缓存与输入无关的核，需固定模型、设备及精度且无状态推理；已核对与未缓存输出一致。它是运行优化，不是新的表征方法。

实验回答的是固定外部预训练表征迁移到睡眠EEG的可行性和当前设置下的性能，不区分扩散训练目标、外部预训练数据、结构或模型容量各自的贡献。EOG提取和跨模态融合属于D负责范围，本实验不声称已完成。
'''
    write(OUT/'EEGDM理解笔记.md',note)

    # Standalone scientific figures, with the exact tables retained beside them.
    result=pd.read_csv(FOLLOW/'comparison_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    xx=np.arange(3)
    for ax,key,title in [(axes[0],'macro_f1','Macro-F1'),(axes[1],'n1_f1','N1 F1')]:
        ax.bar(xx-.18,result['CNN_'+key]*100,.36,label='CNN')
        ax.bar(xx+.18,result['E3_'+key]*100,.36,label='Frozen EEGDM + linear')
        ax.set(xticks=xx,xticklabels=['2024 (primary)','2025','2026'],ylabel=title+' (%)',ylim=(0,100),xlabel='25% subject-list sampling seed')
        ax.legend(fontsize=8)
        for i,(a,b) in enumerate(zip(result['CNN_'+key]*100,result['E3_'+key]*100)):
            ax.text(i-.18,a+1,f'{a:.2f}',ha='center',fontsize=8)
            ax.text(i+.18,b+1,f'{b:.2f}',ha='center',fontsize=8)
    fig.suptitle('Validation results: 8 subjects, 21,186 epochs; training seed 0')
    fig.savefig(OUT/'label25_comparison.png',dpi=170)
    plt.close(fig)
    primary=load(FOLLOW/'metrics_official_E3_25_sseed2024.json')
    full=load(FOLLOW/'metrics_official_E3_100.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    for ax,metric,title in [(axes[0],full,'EEGDM 100% labels'),(axes[1],primary,'EEGDM 25% labels (2024)')]:
        cm=np.array(metric['confusion_matrix'])
        norm=cm/cm.sum(1,keepdims=True)*100
        im=ax.imshow(norm,vmin=0,vmax=100,cmap='Blues')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=['W','N1','N2','N3','REM'],yticklabels=['W','N1','N2','N3','REM'],xlabel='Predicted',ylabel='True',title=title)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{cm[i,j]}\n{norm[i,j]:.1f}%',ha='center',va='center',fontsize=7,color='white' if norm[i,j]>55 else 'black')
    fig.colorbar(im,ax=axes,label='Percent within true stage')
    fig.savefig(OUT/'confusion_primary.png',dpi=170)
    plt.close(fig)
    note_lines=['# 实验③第一周最终复盘与完成清单','', '记录日期：2026-10-23。负责人：C。范围：冻结EEGDM的EEG表征迁移及100%/25%线性分类，不包含D的EOG迁移与融合。',
        '', '## 完成验收','', '| 原任务要求 | 当前证据 | 状态 |','|---|---|---|',
        '| 环境、权重、随机前向 | environment；backbone_source.md；reports/backbone_verification.json | 完成 |',
        '| 一页模型理解笔记 | 本目录EEGDM理解笔记.md | 完成 |',
        '| 转换脚本、输入设置 | scripts/extract_features.py；configs/feature_config.yaml | 完成 |',
        '| 几百段、三条件、两长度 | feature_pilot；本目录adapter_checks.json、channel_pilot_comparison.csv、length_comparison.csv | 完成（候选效果仅小样本） |',
        '| 全量特征与资源/省空间方案 | features/eeg_cond14_v01；本目录resource_summary.json | 完成 |',
        '| 相同名单100%/25%线性分类及CNN对比 | b_followup_20261023；三份B名单，训练seed0；统一评估与重放核验通过 | 完成 |',
        '| 每次运行完整六件套 | 五个run均有README.md、config.yaml、history.csv、best.ckpt、metrics.json、predictions_val.csv | 完成 |',
        '| 1–2个表征假设与复盘 | 见下文 | 完成 |',
        '| D交接材料 | 原20261020全量包＋团队存储20261023最终补充包（生成后见交付校验） | 本地整理完成，发送/接收需由本人确认 |',
        '', '## 结果与现象','',
        '100% EEGDM主方案Macro-F1=74.97%，CNN=76.45%。25%三份名单EEGDM为69.71%、70.18%、72.07%，CNN为71.71%、71.58%、69.15%。均值70.66% vs 70.81%，整体接近，不能认定稳定整体优势。EEGDM的N1 F1三组均更高（40.11/41.45/42.83% vs 38.44/37.58/31.88%）。',
        '', '主名单EEGDM从100%到25%：Macro-F1下降5.25个百分点，N1-F1从46.68%降至40.11%，REM-F1从68.84%降至51.46%。REM是值得优先检查的薄弱环节，但这不直接证明原因。',
        '', '![25%对比](label25_comparison.png)','', '![主方案混淆矩阵](confusion_primary.png)',
        '', '## 输入适配核对','',
        '全部296段、三个条件均核对30秒原生调用和显式六段调用；两者当前实质相同。不能将其作为连续30秒上下文与5秒上下文的科学性能对照。',
        '', '| 条件 | 六人留一人试验的合并OOF Macro-F1 |','|---|---:|']
    for r in adapters['channel_comparison']:
        note_lines.append(f"| {r['condition']} | {r['pooled_oof_macro_f1']*100:.2f}% |")
    note_lines += ['',f"这一小样本检查中得分最高的是条件{adapters['best_on_this_pilot']}。只用原训练集六名受试者、296段、类别平衡抽样，StandardScaler与LogisticRegression均在每折训练部分拟合。这是补充性描述，不使用验证集或测试集，也不能证明全量最优；正式条件14不回溯更改。",
        '', '## 两个待验证的表征假设','',
        '**假设1：冻结EEGDM对N1具有可用的迁移信息，但当前代理条件与简单线性头对REM分离不足。** 证据：三份25%名单N1-F1均高于CNN，主名单REM下降较大。下一阶段可在训练集内部按受试者嵌套交叉验证比较条件与正则化，并以N1/REM混淆为预设观察项；最终仅在独立数据上验证，不反复挑验证分数。当前证据不能归因于扩散目标或特定电极条件。',
        '', '**假设2：简单保留全部时间位置会增大估计与优化难度，时间聚合和正则化可能更适合当前样本量。** 证据：时间展开头384005参数，均值头12805参数；同超参下Macro-F1=68.42% vs74.97%。下一阶段用训练集内部交叉验证调正则化、学习率，并比较5秒分块均值、1秒展开与等维度投影；控制参数量后再判断时间信息是否有益。当前不能声称时间信息无用，也不能仅凭训练曲线断言过拟合。',
        '', '## 资源和省空间建议','',
        f"122记录、154124段，特征{resources['feature_bytes']/1e9:.2f}GB；记录级提取与写盘耗时合计{resources['extraction_compute_and_write_seconds']/3600:.2f}小时，峰值显存{resources['peak_gpu_mb']:.1f}MiB。该计时为各记录之和，不含初始化与记录间索引写入。",
        '', f"float16已实测相对L2误差约0.021%，相较float32约节省一半；只保留均值池化float32特征约{resources['cost_of_mean_pooled_float32_bytes']/1e9:.2f}GB，但会丢失时间结构，D需要完整特征时不删除原始特征。5秒时间窗或选部分层可继续减小体积，尚未正式验证效果。",
        '', '## 障碍、偏差及状态边界','',
        '已解决：WSL环境、S4回退、磁盘容量与B材料依赖。未安装S4自定义扩展。JSON/pt原文件已补导出为团队要求的YAML/CSV/ckpt标准命名；没有因此重训或改分。',
        '', '原文档的PKL是官方输入约定，本机采用直接读取A的NPZ并调用官方提取器，实际张量转换与索引已检查，未为154124段各生成独立PKL。t=1与不加输入随机噪声的区别已在笔记中说明。通道14为预指定代理条件，未在全量三个条件中证明最优，不能将小样本排行说成全量最优。',
        '', '只有一个固定训练种子，三份名单反映的波动与训练随机性不可完全分离；仅八位验证受试者，模型由验证集早停并比较。外部预训练数据重叠未审计。CNN与EEGDM架构、输入变换、优化器和设备不同。全部为第一周迁移可行性证据，测试集未使用。',
        '', '本地实验、核验与交付文件已完成；本次未代发文件给D、未代为上传GitHub或网盘，也不声称D已接收。EOG、融合和全组N1错误分析由D负责，不列为C未完成实验。',
        '', '## 复现与文件入口','',
        '运行索引：run_index.csv；环境与主干来源：../../environment、../../backbone_source.md；实际提取配置：../../configs/feature_config.yaml；25%流程：../../scripts/run_b_followup.py（已完成run会跳过训练并复核）；完整参数与单run命令在各runs/*/README.md。']
    note_text='\n'.join(note_lines)+'\n'
    write(OUT/'实验3_第一周最终复盘.md',note_text)
    dump(OUT/'completion_audit.json',dict(status='complete',scope='C week1 EEG-only experiment',
        completed_runs=names,official_evaluator='B v1',adapter_checks_passed=True,
        test_used=False,external_delivery_confirmed=False,
        d_eog_and_fusion_not_part_of_c_scope=True))
    print(json.dumps(dict(status='complete',runs=names,report=str(OUT/'实验3_第一周最终复盘.md')),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
