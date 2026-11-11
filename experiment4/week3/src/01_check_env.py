"""第 1 步：环境与数据检查（约 1–2 分钟）。

检查：依赖版本、CUDA、官方代码路径、名单 SHA、训练池 NPZ 是否齐全、各候选小模型参数量，
以及每个候选模型在 GPU 上能否完成一次前向 + 反向（batch 4，5 秒输入）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import w3d_common as C


def main():
    import torch
    import diffusers
    import ema_pytorch  # noqa: F401
    import scipy
    print(f'python {sys.version.split()[0]}  torch {torch.__version__}  diffusers {diffusers.__version__}  '
          f'scipy {scipy.__version__}  numpy {np.__version__}', flush=True)
    if not torch.cuda.is_available():
        C.fail('CUDA not available')
    print(f'GPU: {torch.cuda.get_device_name(0)}  '
          f'total memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB', flush=True)

    C.add_vendor_to_path()
    print(f'vendor EEGDM: {C.VENDOR}', flush=True)
    C.check_manifests()
    print('manifest SHA256 OK (split_subjects.csv, label_subsets_v2.csv)', flush=True)

    recs = C.split_records()
    train = sorted(r for r, (_, s) in recs.items() if s == 'train')
    subj = sorted({recs[r][0] for r in train})
    print(f'train split: {len(subj)} subjects, {len(train)} records', flush=True)
    if len(subj) != 54:
        C.fail('Expected 54 training subjects')
    missing = [r for r in train if not (C.DATA / f'{r}.npz').exists()]
    if missing:
        C.fail(f'Missing NPZ files: {missing[:5]} ...')
    with np.load(C.DATA / f'{train[0]}.npz', allow_pickle=False) as z:
        keys = sorted(z.files)
        sig = z['signal']
        print(f'example {train[0]}: keys={keys} signal shape={sig.shape} dtype={sig.dtype}', flush=True)
        if sig.ndim != 3 or sig.shape[1:] != (2, 3000):
            C.fail('Unexpected signal shape; expected (n, 2, 3000)')
        print(f'  EEG ch0 |uV| p99 = {np.percentile(np.abs(sig[:, 0]), 99):.1f}, '
              f'EOG ch1 |uV| p99 = {np.percentile(np.abs(sig[:, 1]), 99):.1f}', flush=True)
    print(f'label25 (14-subject) list: {C.label25_subjects()}', flush=True)

    sch = C.make_scheduler()
    print(f'scheduler: {type(sch).__name__} T={sch.config.num_train_timesteps} '
          f'prediction={sch.config.prediction_type}', flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for name in C.MODEL_CONFIGS:
        model = C.build_model(name, seed=0).cuda()
        n = C.count_params(model)
        x = torch.randn(4, 1, C.WIN, device='cuda')
        t = torch.randint(0, 50, (4, 1), device='cuda')
        cond = torch.tensor([[0], [1], [0], [1]], device='cuda')
        torch.cuda.reset_peak_memory_stats()
        out = model(x, t, cond)
        loss = out.pow(2).mean() + 0 * sum(p.sum() for p in model.parameters())
        loss.backward()
        ok = out.shape == x.shape and torch.isfinite(out).all().item()
        print(f'model {name:13s} params={n:>10,d}  out={tuple(out.shape)}  finite={ok}  '
              f'peak={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB', flush=True)
        if not ok:
            C.fail(f'Forward check failed for {name}')
        del model, out, loss
        torch.cuda.empty_cache()
    print('CHECK PASSED', flush=True)


if __name__ == '__main__':
    main()
