#flips.py —— 翻转统计：比较"单独用EEG/EOG时对不对"和"融合后对不对"
#统计融合到底是帮忙还是帮倒忙，并按睡眠阶段、按受试者分组看

import pandas as pd
import sys
from pathlib import Path

#数据文件夹：本脚本上一级文件夹里的real_data（不管从哪个目录运行都能找到）
DATA = Path(__file__).resolve().parents[1] / "real_data"

#读入fuse.py导出的表（已经包含y_pred_eeg, y_pred_eog, y_pred_fused, y_true等列）
merged = pd.read_csv(DATA / "fused_predictions_val.csv")

print(merged.columns.tolist())

print(f"Loaded rows: {len(merged)}")

#五个阶段的数字标签对应的名字，顺序和全组约定一致：0=W,1=N1,2=N2,3=N3,4=REM
#用来把y_true这一列的数字，翻译回人看得懂的阶段名字，方便后面分组统计
label_to_stage = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

#新增一列stage_name，把数字标签翻译成阶段名字，后面按阶段分组时要用
merged["stage_name"] = merged["y_true"].map(label_to_stage)

#核对一下：翻译后有没有出现"翻译失败"的情况（比如y_true里混进了0-4以外的数字）
#.isna()会把翻译失败的格子标成True(因为字典里查不到对应的值，map会返回缺失值)
if merged["stage_name"].isna().sum() > 0:
    print(f"WARNING: {merged['stage_name'].isna().sum()} rows have y_true outside 0-4, cannot map to stage name")
    sys.exit(1)

#算出三列True/False，分别表示"这一行EEG判断对不对"、"EOG对不对"、"融合后对不对"
correct_eeg = (merged["y_pred_eeg"] == merged["y_true"])
correct_eog = (merged["y_pred_eog"] == merged["y_true"])
correct_fused = (merged["y_pred_fused"] == merged["y_true"])


#定义一个函数，专门用来把"单独判断对不对"和"融合后对不对"这两列True/False
#分类成四种情况，并统计每种情况的数量和占比
#参数说明：
#   correct_ref —— 作为参照的单模态判断对不对（比如correct_eeg）
#   correct_fused —— 融合后判断对不对
#   ref_name —— 参照模态的名字，只是用来打印和存表时标注清楚（比如"EEG"）
def compute_flip_stats(correct_ref, correct_fused, ref_name):
    #四种情况，分别用True/False的组合筛出对应的行
    #参照对、融合错 —— 融合帮倒忙
    ref_correct_fused_wrong = correct_ref & (~correct_fused)
    #参照错、融合对 —— 融合帮了忙
    ref_wrong_fused_correct = (~correct_ref) & correct_fused
    #两边都对 —— 没变化
    both_correct = correct_ref & correct_fused
    #两边都错 —— 没变化
    both_wrong = (~correct_ref) & (~correct_fused)

    #分母是总行数，每个比例都要用这个分母除
    total = len(correct_ref)

    #把结果整理成一个列表，每个元素是一行要存进表格的记录
    #用字典的形式，方便后面转换成DataFrame
    rows = [
        {
            "reference": ref_name,
            "category": "hurt(帮倒忙)",
            "count": int(ref_correct_fused_wrong.sum()),
            "denominator": total,
            "percentage": round(ref_correct_fused_wrong.sum() / total * 100, 2),
        },
        {
            "reference": ref_name,
            "category": "helped(帮了忙)",
            "count": int(ref_wrong_fused_correct.sum()),
            "denominator": total,
            "percentage": round(ref_wrong_fused_correct.sum() / total * 100, 2),
        },
        {
            "reference": ref_name,
            "category": "both_correct(没变-都对)",
            "count": int(both_correct.sum()),
            "denominator": total,
            "percentage": round(both_correct.sum() / total * 100, 2),
        },
        {
            "reference": ref_name,
            "category": "both_wrong(没变-都错)",
            "count": int(both_wrong.sum()),
            "denominator": total,
            "percentage": round(both_wrong.sum() / total * 100, 2),
        },
    ]
    return rows


#以EEG为参照算一遍，再以EOG为参照算一遍
#两次调用上面定义的函数，结果分别存在两个列表里
stats_vs_eeg = compute_flip_stats(correct_eeg, correct_fused, "EEG")
stats_vs_eog = compute_flip_stats(correct_eog, correct_fused, "EOG")

#把两组结果（各4行）合并成一张表，一共8行
#+ 是列表拼接，把两个列表接成一个长列表
all_stats = pd.DataFrame(stats_vs_eeg + stats_vs_eog)

print("\n=== Overall flip stats ===")
print(all_stats.to_string(index=False))

#保存总体翻转统计表
all_stats.to_csv(DATA / "fusion_changes.csv", index=False)
print("\nSaved: real_data/fusion_changes.csv")


#---------- 按阶段、按受试者分组看 ----------

#groupby()是pandas里"分组"的功能，按subject_id和stage_name两列同时分组
#意思是：把所有行按"这个人+这个阶段"的组合切成一个个小组，分别统计
#比如 SC406 + N1 是一组，SC406 + N2 是另一组，分开算
grouped_rows = []

#.groupby([...])返回一个"分组对象"，for循环遍历它时，
#每一次会拿到:这一组的分组标签(subject, stage)，以及属于这一组的那部分数据(group)
for (subject, stage), group in merged.groupby(["subject_id", "stage_name"]):
    #组内这一行行数,也就是这个人在这个阶段一共有多少个30秒片段
    n = len(group)

    #组内三种判断对不对的True/False列(只取这一组的部分)
    g_correct_eeg = (group["y_pred_eeg"] == group["y_true"])
    g_correct_fused = (group["y_pred_fused"] == group["y_true"])

    #以EEG为参照，算这个人这个阶段的融合变化情况
    hurt = (g_correct_eeg & (~g_correct_fused)).sum()
    helped = ((~g_correct_eeg) & g_correct_fused).sum()

    #顺手算一下这个人这个阶段，EEG单独的准确率和融合后的准确率，方便直接对比
    eeg_acc = g_correct_eeg.mean()
    fused_acc = g_correct_fused.mean()

    grouped_rows.append({
        "subject_id": subject,
        "stage_name": stage,
        "n_segments": n,
        "eeg_accuracy": round(eeg_acc, 4),
        "fused_accuracy": round(fused_acc, 4),
        "hurt_count": int(hurt),
        "helped_count": int(helped),
    })

#把按组整理好的结果，转换成一张表
by_subject_stage = pd.DataFrame(grouped_rows)

#按阶段名字、受试者编号排序，方便看的时候N1这个阶段的几行会排在一起
by_subject_stage = by_subject_stage.sort_values(["stage_name", "subject_id"])

print("\n=== Per-subject, per-stage breakdown (first 10 rows) ===")
print(by_subject_stage.head(10).to_string(index=False))

by_subject_stage.to_csv(DATA / "by_subject_stage.csv", index=False)
print("\nSaved: real_data/by_subject_stage.csv")


#---------- 单独把N1这个阶段拎出来看一眼，因为这是最关心的阶段 ----------

n1_rows = by_subject_stage[by_subject_stage["stage_name"] == "N1"]
print("\n=== N1 stage only ===")
print(n1_rows.to_string(index=False))