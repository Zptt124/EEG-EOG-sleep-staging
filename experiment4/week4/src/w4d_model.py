"""W4 backbone：在官方 Wavenet 上加两个可选开关，默认关闭时与第三周模型完全相同。

1) self-conditioning（学长说的 representation feedback；论文 arXiv 2505.10999，Xiang et al.，
   "Conditioning Residuals"，single-shot 版本）：
   - 取第 m 个 block 输出的残差流 h_m [B, L, d_model]，做无参数池化 s_m = pool(h_m)（论文是全局平均）；
   - Δe = P(s_m) ⊙ φ(t)：P 是一层零初始化的 Linear（-> d_cond），φ 是一个单独的 timestep embedder；
   - 把 Δe 加到第 m 个 block 之后所有 block 的全局 condition 上（e~_l = e + Δe, l > m）；
   - 单次前向、无 stop-gradient、无随机丢弃（论文未提）；训练和采样流程不变。
   与论文的差异：这里额外提供 std / mean+std 池化选项（第三周发现 std pooling 读出更好），默认按论文用 mean。

2) cross-modal EEG 条件（W4 跨模态对照，按 C 的规则 experiment3/week4/cross_modal_protocol.md）：
   - 只对 EOG 目标窗（modality=1）给 EEG 条件；EEG 目标窗永远是空条件。
   - EEG 条件窗经 Linear(1,d)-ReLU-Linear(d,d)（第二层零初始化）后逐点加到输入嵌入上（与目标窗时间对齐）；
   - 另有一个“是否有条件”的 flag 嵌入（零初始化）加到全局 condition 上，用来区分“空条件”和“EEG 全为 0”。
   - 空条件（condition dropout、没有错时候选、推理/提特征时只给 EOG）走同一个分支：EEG 分支整体乘 0 且 flag=0。
   - 接口：condition_eeg=None 表示整批都是空条件；condition_mask [B] 是批内逐窗有效掩码（1=有条件，0=空条件）。

构造顺序：先按 seed 构建官方 Wavenet（参数初始化与第三周逐位相同），再构建新增模块。
"""
import torch
from torch import nn
from einops import rearrange

import w4d_common as C


def pool_time(h, how):
    """h [B, L, D] -> [B, D] 或 [B, 2D]"""
    if how == 'mean':
        return h.mean(dim=1)
    if how == 'std':
        return h.std(dim=1)
    if how == 'meanstd':
        return torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)
    raise ValueError(how)


def build_model(cfg_name, seed, self_cond_layer=0, self_cond_pool='mean', cross_modal='none'):
    torch.manual_seed(seed)
    C.add_vendor_to_path()
    from model.diffusion_model import Wavenet
    m = Wavenet(**C.model_kwargs(cfg_name))   # 官方类；前向统一走下面的 w4_forward
    _add_extras(m, self_cond_layer, self_cond_pool, cross_modal)
    return m


def _add_extras(m, self_cond_layer, self_cond_pool, cross_modal):
    from model.util import calc_diffusion_step_embedding
    m._t_embed_fn = calc_diffusion_step_embedding
    n_layer = len(m.layers)
    d_model = m.d_model
    d_cond = m.t_embed_in[0].out_features
    m.self_cond_layer = int(self_cond_layer)
    m.self_cond_pool = self_cond_pool
    m.cross_modal = cross_modal
    if m.self_cond_layer:
        if not (1 <= m.self_cond_layer < n_layer):
            C.fail(f'--self-cond-layer must be in [1, {n_layer - 1}] (1-based block index), got {self_cond_layer}')
        d_in = d_model * (2 if self_cond_pool == 'meanstd' else 1)
        proj = nn.Linear(d_in, d_cond)
        proj.weight.data.zero_()
        proj.bias.data.zero_()
        m.sc_proj = proj
        m.sc_t_embed = nn.Sequential(nn.Linear(m.d_embed, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))
    if cross_modal != 'none':
        last = nn.Linear(d_model, d_model)
        last.weight.data.zero_()
        last.bias.data.zero_()
        m.xc_in = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), last)
        m.xc_flag = nn.Embedding(2, d_cond)
        m.xc_flag.weight.data.zero_()


def w4_forward(m, x, t, cls, condition_eeg=None, condition_mask=None, return_feedback=False):
    """x [B,1,L]，t [B,1]，cls [B,1]；condition_eeg [B,1,L] EEG 条件（None = 全部空条件），
    condition_mask [B] 1=有条件 0=空条件（None 且给了 condition_eeg 时视为全部有条件）。

    return_feedback=True 时额外返回 self-conditioning 的池化特征 s_m [B, D]（下游 probe 用）。
    """
    cond = m.calc_cond(t, cls)                       # [B,1,d_cond]
    h = m.in_layer(rearrange(x, 'B C L -> B L C'))
    if m.cross_modal != 'none':
        B = x.shape[0]
        if condition_eeg is None:
            mask = torch.zeros(B, device=x.device)
        else:
            mask = condition_mask.float() if condition_mask is not None else torch.ones(B, device=x.device)
            h = h + m.xc_in(rearrange(condition_eeg, 'B C L -> B L C')) * mask[:, None, None]
        cond = cond + m.xc_flag(mask.long())[:, None, :]
    all_skip = 0
    feedback = None
    for i, layer in enumerate(m.layers, start=1):
        h, skip = layer(h, cond, None)
        all_skip = all_skip + skip
        if m.self_cond_layer and i == m.self_cond_layer:
            feedback = pool_time(h, m.self_cond_pool)
            phi = m.sc_t_embed(m._t_embed_fn(t, m.d_embed))          # [B, d_cond]
            cond = cond + (m.sc_proj(feedback) * phi)[:, None, :]
    for l in m.out_layer:
        all_skip = l(all_skip)
    out = rearrange(all_skip, 'B L C -> B C L')
    return (out, feedback) if return_feedback else out
