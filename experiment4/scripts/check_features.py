#check_features.py —— 检查EEGDM提取出来的特征文件数值是否正常
#用法：python check_features.py 特征文件.npy

import sys
import numpy as np

#sys.argv是运行时命令行上写的参数列表，sys.argv[0]是脚本自己的名字，sys.argv[1]是后面跟的第一个参数
#这样同一个脚本可以检查任意一个特征文件，不用每次改代码
path = sys.argv[1]

#np.load读取.npy文件
#mmap_mode="r"：不一次性全部读进内存，用到哪部分才读哪部分。全量特征一个文件可能几百MB，这样更省内存
feat = np.load(path, mmap_mode="r")

print(f"File: {path}")
#shape是数组每个维度的长度，应该是(样本数, 20层, 30个时间窗, 128维)
print(f"Shape: {feat.shape}")
#dtype是数据类型，C说保存时用float16（半精度）
print(f"Dtype: {feat.dtype}")

#检查形状：后三个维度必须是20、30、128，否则不是我们要的格式，直接停
if feat.ndim != 4 or feat.shape[1:] != (20, 30, 128):
    print("WARNING: shape is not (N, 20, 30, 128)")
    sys.exit(1)

#转成float32再算统计量。float16精度低、能表示的最大数只有约65504，直接求和很容易溢出成无穷大
x = np.asarray(feat, dtype=np.float32)
n = x.shape[0]

#---------- 检查1：NaN和无穷大 ----------
#np.isnan(x)对每个数判断是不是NaN，得到一个同样形状的True/False数组，.sum()数出有几个True
n_nan = int(np.isnan(x).sum())
n_inf = int(np.isinf(x).sum())
print(f"NaN count: {n_nan}")
print(f"Inf count: {n_inf}")

#---------- 检查2：整段全是0的样本 ----------
#x.reshape(n, -1)：把每个样本的20x30x128个数拉平成一行（-1表示"剩下的自动算"）
#np.abs(...).max(axis=1)：每个样本里绝对值最大的那个数；如果它都是0，说明这个样本全是0
flat = x.reshape(n, -1)
all_zero = int((np.abs(flat).max(axis=1) == 0).sum())
print(f"All-zero samples: {all_zero} / {n}")

#---------- 检查3：整体数值范围 ----------
print(f"Overall min / max: {flat.min():.4f} / {flat.max():.4f}")
print(f"Overall mean / std: {flat.mean():.4f} / {flat.std():.4f}")

#---------- 检查4：每一层的情况，以及样本之间有没有差别 ----------
print("\nPer layer:")
print("layer   mean      std   between_sample_std")
for layer in range(20):
    #取出这一层的所有数据，形状(n, 30, 128)
    lx = x[:, layer]
    #between_sample_std：先把每个样本在这一层的特征按时间平均（得到n个128维的向量），
    #再看这n个样本之间差别有多大。如果接近0，说明这一层对所有样本输出几乎一样，区分不了睡眠阶段
    per_sample = lx.mean(axis=1)          #(n, 128)
    between = per_sample.std(axis=0).mean()  #每一维在样本之间的标准差，再对128维取平均
    print(f"{layer:5d} {lx.mean():8.4f} {lx.std():8.4f} {between:12.4f}")

#---------- 结论 ----------
problems = []
if n_nan > 0 or n_inf > 0:
    problems.append("contains NaN/Inf")
if all_zero > 0:
    problems.append("has all-zero samples")
if problems:
    print("\nRESULT: PROBLEM - " + ", ".join(problems))
else:
    print("\nRESULT: OK - no NaN/Inf, no all-zero samples")