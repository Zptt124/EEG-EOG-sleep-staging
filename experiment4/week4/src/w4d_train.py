"""W4 训练核心：Trainer（分模态 loss、self-cond / cross-modal 开关）+ ValSet（held-out 分段 val MSE）。

训练目标与官方 PLDiffusionModel.training_step、第三周 w3d_train.py 相同：
  t ~ Uniform{0..49}，noise ~ N(0,1)，x_t = add_noise(x0, noise, t)，
  目标 = v（v_prediction），loss = MSE(model(x_t, t, modality), v)，梯度裁剪 norm 1，EMA 0.999。
新增：
  - 每个 batch 的 loss 按模态拆开（loss_eeg / loss_eog，只记录，不改变优化目标）；
  - val MSE（W4 共同规则 2）：held-out 8 人上固定窗口、固定噪声、固定 t，t 每 5 步一段，EEG/EOG 分开。
"""
import numpy as np
import torch

import w4d_common as C
from w4d_model import build_model, w4_forward

T_STEPS = C.NOISE_SCH_KWARGS['num_train_timesteps']   # 50
BUCKET = 5
N_BUCKET = T_STEPS // BUCKET                            # 10 段：[0,5) ... [45,50)
STOP_BUCKETS = (1, 2, 3, 4)                             # 中小 t：t ∈ [5, 25)
VAL_SEED = 20261114
VAL_PER_SUBJECT = 128                                   # 每人每模态 128 个 5 秒窗 -> 每模态 1024 个


class ValSet:
    """固定的 held-out 验证集：同一批窗口、同一份噪声、同一组 t，保证不同 checkpoint、不同 run 可比。

    每个窗口在 10 个 t 段里各评一次（段内 t 固定随机抽一个），所以每格（模态×t 段）有 1024 个样本。
    选窗、t、噪声只依赖 held-out 名单和 VAL_SEED，与模型、训练 seed、开关无关。
    """

    def __init__(self, pool, heldout, device='cuda', cross_modal='none', pairs=None, per_subject=VAL_PER_SUBJECT):
        rng = np.random.default_rng(VAL_SEED)
        units = []
        for s in sorted(heldout):
            rows = np.nonzero(pool.subject == s)[0]
            if len(rows) == 0:
                C.fail(f'Held-out subject {s} not found in cache')
            for m in (0, 1):
                nw = pool.n_win                                  # 5 秒 = 6；连续 30 秒 = 1
                cand = (rows[:, None] * 2 * nw + m * nw + np.arange(nw)[None, :]).reshape(-1)
                units.append(np.sort(rng.choice(cand, size=per_subject, replace=False)))
        self.units = np.concatenate(units)
        n = len(self.units)
        t = np.arange(N_BUCKET)[None, :] * BUCKET + rng.integers(0, BUCKET, size=(n, N_BUCKET))
        x, m = pool.gather(self.units)
        g = torch.Generator().manual_seed(VAL_SEED)
        noise = torch.randn((n, N_BUCKET, 1, pool.win), generator=g)
        self.win = pool.win
        self.device = device
        self.x = torch.from_numpy(x).to(device)
        self.m = torch.from_numpy(m).to(device)
        self.t = torch.from_numpy(t).to(device)
        self.noise = noise.to(device)
        self.xc = self.xc_valid = None
        if cross_modal != 'none':
            xc, valid = pool.gather_eeg_cond(self.units, cross_modal, pairs)
            self.xc = torch.from_numpy(xc).to(device)
            self.xc_valid = torch.from_numpy(valid).to(device)
        self.n = n
        import hashlib
        sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        self.t_np = t
        self.fingerprint = dict(n_windows=int(n), val_seed=VAL_SEED, per_subject_per_modality=per_subject,
                                bucket=BUCKET, n_bucket=N_BUCKET,
                                units_sha256=sha(self.units.astype('<i8')),             # [n] int64 单元编号
                                t_sha256=sha(t.astype('<i8')),                         # [n, 10] int64
                                noise_sha256=sha(noise.numpy().astype('<f4')),         # [n, 10, 1, 1000] float32
                                noise_recipe=f'torch.Generator(cpu).manual_seed({VAL_SEED}); '
                                             f'torch.randn(({n}, {N_BUCKET}, 1, {pool.win})), float32')
        if pool.window_sec != 5:   # 5 秒的指纹保持原样（与 W4 各 run 的 config.json 逐字相同）
            self.fingerprint['window_sec'] = pool.window_sec

    @torch.no_grad()
    def evaluate(self, net, sch, with_eeg_cond=False, batch=512):
        """返回 per [n, 10] 的逐样本 MSE（CPU numpy）。with_eeg_cond 只对 cross-modal 模型有意义：
        False = 空条件（只给 EOG，这是主口径，所有 run 可比）；True = 给 EEG 条件（仅 EOG 目标窗）。"""
        was = net.training
        net.eval()
        batch = max(1, batch * 1000 // self.win)   # 30 秒窗每个样本长 6 倍，按信号量缩小 batch；5 秒时不变
        idx_w = torch.arange(self.n, device=self.device).repeat_interleave(N_BUCKET)
        idx_b = torch.arange(N_BUCKET, device=self.device).repeat(self.n)
        out = torch.empty(self.n * N_BUCKET, device=self.device)
        for s in range(0, len(idx_w), batch):
            w, b = idx_w[s:s + batch], idx_b[s:s + batch]
            x0, m, t, noise = self.x[w], self.m[w], self.t[w, b][:, None], self.noise[w, b]
            xt = sch.add_noise(x0, noise, t)
            v = sch.get_velocity(x0, noise, t)
            xc = mask = None
            if with_eeg_cond and self.xc is not None:
                xc, mask = self.xc[w], ((m[:, 0] == 1) & self.xc_valid[w]).float()
            pred = w4_forward(net, xt, t, m, condition_eeg=xc, condition_mask=mask)
            out[s:s + batch] = ((pred - v) ** 2).mean(dim=(1, 2))
        net.train(was)
        return out.view(self.n, N_BUCKET).cpu().numpy()

    def summarize(self, per):
        """-> dict：{eeg,eog}_t{lo}_{hi}、{eeg,eog}_all、stop_metric（EEG/EOG 各自 t∈[5,25) 平均后再等权平均）。"""
        m = self.m[:, 0].cpu().numpy()
        out = {}
        stop = []
        for mi, name in enumerate(C.MOD_NAMES):
            pm = per[m == mi]
            for b in range(N_BUCKET):
                out[f'{name}_t{b * BUCKET:02d}_{(b + 1) * BUCKET:02d}'] = float(pm[:, b].mean())
            out[f'{name}_all'] = float(pm.mean())
            out[f'{name}_t05_25'] = float(pm[:, list(STOP_BUCKETS)].mean())
            stop.append(out[f'{name}_t05_25'])
        out['stop_metric'] = float(np.mean(stop))
        return out


class Trainer:
    def __init__(self, cfg_name, pool, units, batch_size, seed, lr=None, device='cuda',
                 self_cond_layer=0, self_cond_pool='mean', cross_modal='none', cond_dropout=0.0, pairs=None):
        from ema_pytorch import EMA
        self.cfg_name, self.pool, self.units = cfg_name, pool, np.asarray(units, dtype=np.int64)
        self.batch_size, self.seed, self.device = batch_size, seed, device
        self.cross_modal, self.cond_dropout, self.pairs = cross_modal, cond_dropout, pairs
        self.model = build_model(cfg_name, seed, self_cond_layer, self_cond_pool, cross_modal).to(device)
        self.ema = EMA(self.model, **C.EMA_KWARGS).to(device)
        self.opt = C.make_optimizer(self.model)
        if lr is not None:   # 只改通用参数组；S4 参数组保持官方设置
            self.opt.param_groups[0]['lr'] = lr
        self.sch = C.make_scheduler()
        torch.manual_seed(seed + 1)
        self.np_rng = np.random.default_rng(seed + 2)        # 打乱顺序
        # condition dropout 用独立随机流（不碰 torch 的噪声/t 随机流），三组 none/same/shifted 的噪声和 t 完全相同
        self.drop_rng = np.random.default_rng(seed + 3)
        self.step, self.epoch, self.pos = 0, 0, 0
        self.perm = self.np_rng.permutation(self.units)

    # ---------- 取下一个 batch（按 training epoch 打乱，走完一轮换新排列） ----------
    def next_units(self):
        if self.pos + self.batch_size > len(self.perm):
            self.epoch += 1
            self.pos = 0
            self.perm = self.np_rng.permutation(self.units)
        u = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return u

    def train_step(self):
        units = self.next_units()
        x, m = self.pool.gather(units)
        x0 = torch.from_numpy(x).to(self.device, non_blocking=True)
        m = torch.from_numpy(m).to(self.device, non_blocking=True)
        xc = mask = None
        if self.cross_modal != 'none':
            xc, valid = self.pool.gather_eeg_cond(units, self.cross_modal, self.pairs)
            keep = self.drop_rng.random(len(units)) >= self.cond_dropout              # 逐窗 condition dropout
            mask = torch.from_numpy(valid & keep).to(self.device) & (m[:, 0] == 1)  # 只有 EOG 目标窗带 EEG 条件
            xc, mask = torch.from_numpy(xc).to(self.device), mask.float()
        bs = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.randint(0, T_STEPS, (bs, 1), device=self.device)
        xt = self.sch.add_noise(x0, noise, t)
        target = self.sch.get_velocity(x0, noise, t)
        pred = w4_forward(self.model, xt, t, m, condition_eeg=xc, condition_mask=mask)
        per = ((pred - target) ** 2).mean(dim=(1, 2))
        loss = per.mean()          # 与 F.mse_loss(pred, target) 相同（每个样本长度相同）
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP)
        self.opt.step()
        self.ema.update()
        self.step += 1
        with torch.no_grad():
            is_eog = m[:, 0] == 1
            # 某个模态在 batch 里没有样本时记 nan，汇总时用 nanmean
            l_eeg = per[~is_eog].mean() if (~is_eog).any() else torch.tensor(float('nan'), device=self.device)
            l_eog = per[is_eog].mean() if is_eog.any() else torch.tensor(float('nan'), device=self.device)
        return torch.stack([loss.detach(), l_eeg, l_eog, gnorm.detach()])

    # ---------- 断点续训 ----------
    def state(self):
        return dict(cfg_name=self.cfg_name, seed=self.seed, step=self.step, epoch=self.epoch, pos=self.pos,
                    perm=self.perm, model=self.model.state_dict(), ema=self.ema.state_dict(),
                    opt=self.opt.state_dict(), np_rng=self.np_rng.bit_generator.state,
                    drop_rng=self.drop_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())

    def load_state(self, s):
        if s['cfg_name'] != self.cfg_name or s['seed'] != self.seed:
            C.fail('Resume state does not match config/seed')
        self.model.load_state_dict(s['model'])
        self.ema.load_state_dict(s['ema'])
        self.opt.load_state_dict(s['opt'])
        self.step, self.epoch, self.pos, self.perm = s['step'], s['epoch'], s['pos'], s['perm']
        self.np_rng.bit_generator.state = s['np_rng']
        self.drop_rng.bit_generator.state = s['drop_rng']
        torch.set_rng_state(s['torch_rng'].cpu())
        torch.cuda.set_rng_state(s['cuda_rng'].cpu())

    def weights(self, extra=None):
        """供下游提特征的权重：online 与 EMA 两份（step 0 时二者相同 = random-init）。"""
        d = dict(cfg_name=self.cfg_name, model_kwargs=C.model_kwargs(self.cfg_name), seed=self.seed,
                 step=self.step, epoch=self.epoch, model=self.model.state_dict(),
                 ema_model=self.ema.ema_model.state_dict())
        d.update(extra or {})
        return d
