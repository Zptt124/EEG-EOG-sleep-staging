"""扩散预训练的训练核心：测速（03）和正式短训（04）共用。

训练目标与官方 PLDiffusionModel.training_step 相同：
  t ~ Uniform{0..49}，noise ~ N(0,1)，x_t = add_noise(x0, noise, t)，
  目标 = v（v_prediction），loss = MSE(model(x_t, t, modality), v)，梯度裁剪 norm 1，EMA 0.999。
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import w3d_common as C

DIAG_N = 512
DIAG_SEED = 12345


class Trainer:
    def __init__(self, cfg_name, pool, units, batch_size, seed, lr=None, device='cuda'):
        from ema_pytorch import EMA
        self.cfg_name, self.pool, self.units = cfg_name, pool, np.asarray(units, dtype=np.int64)
        self.batch_size, self.seed, self.device = batch_size, seed, device
        self.model = C.build_model(cfg_name, seed).to(device)
        self.ema = EMA(self.model, **C.EMA_KWARGS).to(device)
        self.opt = C.make_optimizer(self.model)
        if lr is not None:   # 只改通用参数组；S4 参数组保持官方设置
            self.opt.param_groups[0]['lr'] = lr
        self.sch = C.make_scheduler()
        self.T = C.NOISE_SCH_KWARGS['num_train_timesteps']
        torch.manual_seed(seed + 1)
        self.np_rng = np.random.default_rng(seed + 2)
        self.step, self.epoch, self.pos = 0, 0, 0
        self.perm = self.np_rng.permutation(self.units)
        self._make_diag()

    # ---------- 固定诊断集：同一批窗口、同一份噪声、固定 t ----------
    def _make_diag(self):
        rng = np.random.default_rng(DIAG_SEED)
        u = np.sort(rng.choice(self.pool.n_units, size=DIAG_N, replace=False))
        x, m = self.pool.gather(u)
        g = torch.Generator().manual_seed(DIAG_SEED)
        self.diag = dict(units=u, x=torch.from_numpy(x).to(self.device), m=torch.from_numpy(m).to(self.device),
                         noise=torch.randn(x.shape, generator=g).to(self.device),
                         t=torch.from_numpy(np.arange(DIAG_N) % self.T)[:, None].to(self.device))

    @torch.no_grad()
    def diag_loss(self, use_ema=False):
        net = self.ema.ema_model if use_ema else self.model
        was = net.training
        net.eval()
        d = self.diag
        out = {}
        losses = []
        for s in range(0, DIAG_N, 128):
            sl = slice(s, s + 128)
            xt = self.sch.add_noise(d['x'][sl], d['noise'][sl], d['t'][sl])
            v = self.sch.get_velocity(d['x'][sl], d['noise'][sl], d['t'][sl])
            pred = net(xt, d['t'][sl], d['m'][sl])
            losses.append(((pred - v) ** 2).mean(dim=(1, 2)))
        per = torch.cat(losses)
        t = d['t'][:, 0]
        out['all'] = per.mean().item()
        out['t_lt10'] = per[t < 10].mean().item()
        out['t_10_30'] = per[(t >= 10) & (t < 30)].mean().item()
        out['t_ge30'] = per[t >= 30].mean().item()
        out['eeg'] = per[d['m'][:, 0] == 0].mean().item()
        out['eog'] = per[d['m'][:, 0] == 1].mean().item()
        net.train(was)
        return out

    # ---------- 取下一个 batch（按 training epoch 打乱，走完一轮换新排列） ----------
    def next_units(self):
        if self.pos + self.batch_size > len(self.perm):
            self.epoch += 1
            self.pos = 0
            self.perm = self.np_rng.permutation(self.units)
        u = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return u

    def load_batch(self, units):
        x, m = self.pool.gather(units)
        return (torch.from_numpy(x).to(self.device, non_blocking=True),
                torch.from_numpy(m).to(self.device, non_blocking=True))

    def train_step(self):
        x0, m = self.load_batch(self.next_units())
        bs = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.T, (bs, 1), device=self.device)
        xt = self.sch.add_noise(x0, noise, t)
        target = self.sch.get_velocity(x0, noise, t)
        pred = self.model(xt, t, m)
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP)
        self.opt.step()
        self.ema.update()
        self.step += 1
        return loss.detach(), gnorm.detach()

    def param_vector(self):
        return torch.cat([p.detach().flatten().float().cpu() for p in self.model.parameters()])

    # ---------- 断点续训 ----------
    def state(self):
        return dict(cfg_name=self.cfg_name, seed=self.seed, step=self.step, epoch=self.epoch, pos=self.pos,
                    perm=self.perm, model=self.model.state_dict(), ema=self.ema.state_dict(),
                    opt=self.opt.state_dict(), np_rng=self.np_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())

    def load_state(self, s):
        if s['cfg_name'] != self.cfg_name or s['seed'] != self.seed:
            C.fail('Resume state does not match config/seed')
        self.model.load_state_dict(s['model'])
        self.ema.load_state_dict(s['ema'])
        self.opt.load_state_dict(s['opt'])
        self.step, self.epoch, self.pos, self.perm = s['step'], s['epoch'], s['pos'], s['perm']
        self.np_rng.bit_generator.state = s['np_rng']
        torch.set_rng_state(s['torch_rng'].cpu())
        torch.cuda.set_rng_state(s['cuda_rng'].cpu())

    def weights(self):
        """供下游提特征的权重：online 与 EMA 两份（step 0 时二者相同 = random-init）。"""
        return dict(cfg_name=self.cfg_name, model_kwargs=C.model_kwargs(self.cfg_name), seed=self.seed,
                    step=self.step, epoch=self.epoch, model=self.model.state_dict(),
                    ema_model=self.ema.ema_model.state_dict())


def timed_steps(trainer, n_steps, log_every=25, max_seconds=None, log_fn=print):
    """跑 n_steps 步，返回每步耗时列表和 loss/梯度记录。"""
    times, losses, gnorms = [], [], []
    start = time.perf_counter()
    for i in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, gn = trainer.train_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        gnorms.append(gn.item())
        if not (np.isfinite(losses[-1]) and np.isfinite(gnorms[-1])):
            C.fail(f'Non-finite loss/grad at step {trainer.step}: loss={losses[-1]} grad={gnorms[-1]}')
        if (i + 1) % log_every == 0:
            log_fn(f'  step {trainer.step:6d} loss {np.mean(losses[-log_every:]):.4f} '
                   f'grad {np.mean(gnorms[-log_every:]):.3f} step_time {np.median(times[-log_every:]) * 1000:.0f} ms')
        if max_seconds and time.perf_counter() - start > max_seconds:
            log_fn(f'  time cap {max_seconds}s reached after {i + 1} steps')
            break
    return times, losses, gnorms
