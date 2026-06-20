#!/usr/bin/env python
"""
AbDiff model — AF3-style all-atom-in-R3 coordinate diffusion, conditioned on pLM
embeddings. Self-contained (no OpenFold3 import) but faithful to the EDM
preconditioning / noise schedule used in OF3's diffusion_module.py:

    r        = x_noisy / sqrt(t^2 + sigma_data^2)         # input scaling
    x_out    = c_skip(t) * x_noisy + c_out(t) * F(r,...)  # EDM skip connection
    c_skip   = sigma_data^2 / (sigma_data^2 + t^2)
    c_out    = sigma_data * t / sqrt(sigma_data^2 + t^2)

There is NO SO(3)/frame representation. Global pose freedom is handled by
random-rotation augmentation of the target + a rigid (Kabsch) alignment in the loss,
exactly the AF3 recipe (centre_random_augmentation + weighted rigid align).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BB = 14  # atom14 all-atom representation (AF atom14 layout; CA is index 1).
           # Backbone-only runs used 4; all-atom uses 14. Symbol kept for compatibility.


# ----------------------------- conditioning ---------------------------------
class FourierSigmaEmbed(nn.Module):
    """Embed log-noise-level (AF3 uses Fourier features of 0.25*ln(t/sigma_data))."""
    def __init__(self, dim, sigma_data):
        super().__init__()
        self.sigma_data = sigma_data
        self.register_buffer("w", torch.randn(dim // 2))
        self.register_buffer("b", torch.rand(dim // 2))

    def forward(self, t):                       # t: [B]
        c = 0.25 * torch.log(t / self.sigma_data + 1e-8)
        proj = 2 * math.pi * (c[:, None] * self.w[None] + self.b[None])
        return torch.cat([proj.sin(), proj.cos()], dim=-1)   # [B, dim]


class RelPos(nn.Module):
    """AF3 relative-position encoding with a separate 'different chain' bucket."""
    def __init__(self, c_z, max_rel=32):
        super().__init__()
        self.max_rel = max_rel
        self.lin = nn.Linear(2 * max_rel + 2, c_z)

    def forward(self, residue_index, asym_id):       # [B,N] each
        d = residue_index[:, :, None] - residue_index[:, None, :]
        d = d.clamp(-self.max_rel, self.max_rel) + self.max_rel       # [B,N,N] in [0,2max]
        same = (asym_id[:, :, None] == asym_id[:, None, :])
        onehot = F.one_hot(d.long(), num_classes=2 * self.max_rel + 1).float()
        diff_chain = (~same).float()[..., None]
        feat = torch.cat([onehot * same.float()[..., None], diff_chain], dim=-1)
        return self.lin(feat)                                          # [B,N,N,c_z]


# ----------------------------- transformer ----------------------------------
class PairBiasAttention(nn.Module):
    def __init__(self, c, n_head, c_z):
        super().__init__()
        self.n_head, self.c = n_head, c
        self.hd = c // n_head
        self.ln = nn.LayerNorm(c)
        self.to_qkv = nn.Linear(c, 3 * c, bias=False)
        self.z_ln = nn.LayerNorm(c_z)
        self.z_bias = nn.Linear(c_z, n_head, bias=False)
        self.proj = nn.Linear(c, c)

    def forward(self, x, z, mask):                 # x:[B,N,c] z:[B,N,N,c_z] mask:[B,N]
        B, N, _ = x.shape
        h = self.ln(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, N, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, N, self.n_head, self.hd).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd)
        att = att + self.z_bias(self.z_ln(z)).permute(0, 3, 1, 2)      # [B,H,N,N]
        att = att.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(att.dtype).min)
        att = att.softmax(-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, N, self.c)
        return self.proj(out)


class Transition(nn.Module):
    def __init__(self, c, mult=4):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, c * mult)
        self.fc2 = nn.Linear(c * mult, c)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(self.ln(x))))


class DiffBlock(nn.Module):
    def __init__(self, c, n_head, c_z):
        super().__init__()
        self.attn = PairBiasAttention(c, n_head, c_z)
        self.tran = Transition(c)

    def forward(self, x, z, mask):
        x = x + self.attn(x, z, mask)
        x = x + self.tran(x)
        return x


# ------------------------------- model ---------------------------------------
class AbDiffusion(nn.Module):
    def __init__(self, c_esm=640, c=384, c_z=128, n_block=8, n_head=12,
                 sigma_data=16.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.c = c
        # conditioning from pLM single rep
        self.s_in = nn.Sequential(nn.LayerNorm(c_esm), nn.Linear(c_esm, c))
        self.sigma_embed = FourierSigmaEmbed(c, sigma_data)
        self.relpos = RelPos(c_z)
        # outer-product-mean style pair init from single rep
        self.opm = nn.Linear(c, c_z)
        # noisy-coordinate encoder: 4 atoms x 3 coords per token
        self.coord_in = nn.Linear(N_BB * 3, c)
        self.blocks = nn.ModuleList([DiffBlock(c, n_head, c_z) for _ in range(n_block)])
        self.out_ln = nn.LayerNorm(c)
        self.coord_out = nn.Linear(c, N_BB * 3)
        nn.init.zeros_(self.coord_out.weight); nn.init.zeros_(self.coord_out.bias)

    def make_pair(self, s, residue_index, asym_id):
        z = self.relpos(residue_index, asym_id)
        o = self.opm(s)
        z = z + o[:, :, None, :] + o[:, None, :, :]
        return z

    def forward(self, x_noisy, t, emb, residue_index, asym_id, token_mask):
        """
        x_noisy: [B,N,4,3]   t:[B]   emb:[B,N,c_esm]
        returns x_out: [B,N,4,3]  (denoised prediction, EDM skip applied)
        """
        B, N = emb.shape[:2]
        sd = self.sigma_data
        r = x_noisy / torch.sqrt(t[:, None, None, None] ** 2 + sd ** 2)
        s = self.s_in(emb) + self.sigma_embed(t)[:, None, :]
        z = self.make_pair(s, residue_index, asym_id)
        h = s + self.coord_in(r.reshape(B, N, N_BB * 3))
        for blk in self.blocks:
            h = blk(h, z, token_mask)
        upd = self.coord_out(self.out_ln(h)).reshape(B, N, N_BB, 3)
        c_skip = sd ** 2 / (sd ** 2 + t ** 2)
        c_out = sd * t / torch.sqrt(sd ** 2 + t ** 2)
        x_out = c_skip[:, None, None, None] * x_noisy + c_out[:, None, None, None] * upd
        return x_out


# ------------------------------ diffusion utils ------------------------------
def kabsch_align(P, Q, w):
    """Rigid-align P onto Q (per batch). P,Q:[B,M,3] w:[B,M] weights. Returns P aligned.
    SVD is fp32-only AND autocast re-casts matmul/einsum to fp16, so we must
    DISABLE autocast for this whole block (not just .float() the inputs)."""
    out_dtype = P.dtype
    dev_type = "cuda" if P.is_cuda else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        P, Q, w = P.float(), Q.float(), w.float()
        w = w[..., None]
        wsum = w.sum(1, keepdim=True).clamp_min(1e-6)
        Pc = (P * w).sum(1, keepdim=True) / wsum
        Qc = (Q * w).sum(1, keepdim=True) / wsum
        P0, Q0 = P - Pc, Q - Qc
        H = torch.einsum("bmi,bmj->bij", w * P0, Q0)
        U, _, Vh = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(torch.matmul(Vh.transpose(-1, -2), U.transpose(-1, -2))))
        D = torch.eye(3, device=P.device).expand(P.shape[0], 3, 3).clone()
        D[:, 2, 2] = d
        R = torch.matmul(torch.matmul(Vh.transpose(-1, -2), D), U.transpose(-1, -2))
    return (torch.einsum("bij,bmj->bmi", R, P0) + Qc).to(out_dtype)


def sample_sigma(B, sigma_data, device, P_mean=-1.2, P_std=1.5):
    return (sigma_data * torch.exp(P_mean + P_std * torch.randn(B, device=device)))


def random_rotation(B, device, dtype):
    q = torch.randn(B, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(B, 3, 3)
    return R


def make_noise_schedule(n_steps, device, sigma_data=16.0, s_max=160.0, s_min=4e-4, p=7):
    """OF3/AF3 noise schedule (decreasing sigma, ends near 0)."""
    t = torch.arange(0, n_steps + 1, device=device, dtype=torch.float32) / n_steps
    return sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p


@torch.no_grad()
def sample_backbone(model, emb, residue_index, asym_id, token_mask, atom_mask,
                    n_steps=200, gamma_0=0.8, gamma_min=1.0, noise_scale=1.003,
                    step_scale=1.5):
    """AF3 Algorithm-18-style EDM rollout for backbone atoms (R3, no SO3).
    emb:[B,N,c_esm]; returns x:[B,N,4,3]."""
    B, N = token_mask.shape
    dev = emb.device
    sched = make_noise_schedule(n_steps, dev, sigma_data=model.sigma_data)
    am = atom_mask[..., None].float()
    x = sched[0] * torch.randn(B, N, N_BB, 3, device=dev) * am
    for i in range(1, len(sched)):
        c, prev = sched[i], sched[i - 1]
        gamma = gamma_0 if prev > gamma_min else 0.0
        t_hat = prev * (1 + gamma)
        noise = noise_scale * torch.sqrt(torch.clamp(t_hat ** 2 - prev ** 2, min=0.0)) * torch.randn_like(x)
        x_noisy = x + noise * am
        t_b = torch.full((B,), float(t_hat), device=dev)
        x_den = model(x_noisy, t_b, emb, residue_index, asym_id, token_mask)
        d = (x_noisy - x_den) / t_hat
        x = (x_noisy + step_scale * (c - t_hat) * d) * am
    return x


def edm_loss(model, batch, cdr_weight=1.0, h3_weight=1.0):
    """AF3/EDM diffusion training loss. cdr_weight upweights all CDR residues,
    h3_weight additionally upweights CDR-H3 (the hard loop)."""
    coords = batch["coords"]            # [B,N,4,3]
    amask = batch["atom_mask"].float()  # [B,N,4]
    tmask = batch["token_mask"]         # [B,N]
    B, N = tmask.shape
    dev = coords.device
    sd = model.sigma_data

    # center on CA centroid, then random-rotation augmentation (no SO3 manifold)
    ca = coords[:, :, 1]                                   # [B,N,3]
    w_tok = (tmask * batch["atom_mask"][:, :, 1].float())  # [B,N]
    num = (ca * w_tok[..., None]).sum(1, keepdim=True)      # [B,1,3]
    den = w_tok.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)  # [B,1,1]
    cen = num / den                                        # [B,1,3]
    x0 = coords - cen[:, None, :, :]                       # [B,N,4,3] - [B,1,1,3]
    R = random_rotation(B, dev, x0.dtype)
    x0 = torch.einsum("bij,bnaj->bnai", R, x0)
    x0 = x0 * amask[..., None]

    t = sample_sigma(B, sd, dev)
    noise = t[:, None, None, None] * torch.randn_like(x0)
    x_noisy = x0 + noise * amask[..., None]
    x_hat = model(x_noisy, t, batch["emb"], batch["residue_index"], batch["asym_id"], tmask)

    # rigid-align prediction to target (weighted) before MSE
    flat_w = (amask * tmask[..., None]).reshape(B, N * N_BB)
    x_hat_a = kabsch_align(x_hat.reshape(B, N * N_BB, 3), x0.reshape(B, N * N_BB, 3), flat_w)
    x_hat_a = x_hat_a.reshape(B, N, N_BB, 3)

    w = (t ** 2 + sd ** 2) / (t * sd) ** 2          # EDM weighting [B]
    se = ((x_hat_a - x0) ** 2).sum(-1) * amask      # [B,N,N_atom]
    # per-residue CDR weighting: framework=1, any CDR=cdr_weight, CDR-H3 *= h3_weight
    rw = torch.ones(B, N, device=dev)
    if ("cdr" in batch) and (cdr_weight != 1.0 or h3_weight != 1.0):
        cdr = batch["cdr"]; ht = batch["htype"]
        rw = torch.where(cdr > 0, torch.full_like(rw, cdr_weight), rw)
        rw = torch.where((cdr == 3) & (ht == 1), rw * h3_weight, rw)
    se_w = se * rw[..., None]                        # weight atoms by their residue
    denom = (amask * rw[..., None]).sum((1, 2)).clamp_min(1.0)
    per = se_w.sum((1, 2)) / denom
    loss = (w * per).mean()
    with torch.no_grad():                            # report UNweighted rmsd for comparability
        rmsd = torch.sqrt((se.sum((1, 2)) / amask.sum((1, 2)).clamp_min(1.0))).mean()
    return loss, rmsd
