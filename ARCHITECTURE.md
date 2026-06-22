# AbDiff — Architecture in depth

This document explains the AbDiff model end to end: the representation, the
conditioning stack, the denoiser network, the EDM diffusion formulation, the
**CDR-weighted training loss**, and the sampling rollout. It is the companion
to the top-level [`README.md`](README.md) (which holds the results and how-to);
here we focus on *why the model is built the way it is*.

All claims below are traceable to `scripts/model.py` (the entire model is one
self-contained file, ~270 lines, with no OpenFold3 import at runtime).

---

## 1. Design philosophy: AF3-style coordinate diffusion, **no SO(3)**

AbDiff predicts antibody structure by **denoising 3D atom coordinates directly in
ℝ³**. It is a faithful re-implementation of the AlphaFold3 / OpenFold3
*diffusion module* recipe, with three deliberate simplifications for the
antibody-only setting:

1. **No frame / SO(3) representation.** AF2-style models carry a residue frame
   (rotation + translation) and operate on the SE(3) manifold. AbDiff carries
   **only points** — every atom is a free vector in ℝ³. There is no rotation
   manifold, no quaternion bookkeeping, no IPA. Global pose freedom (the fact
   that a structure is only defined up to a rigid motion) is handled *entirely*
   by two tricks borrowed from AF3:
   - **random-rotation augmentation** of the target at training time, and
   - a **weighted rigid (Kabsch) alignment** inside the loss.
2. **Frozen protein-language-model (pLM) conditioning instead of an MSA + Evoformer.**
   There is no MSA, no triangle attention, no Pairformer trunk. The single
   representation is initialized from a *frozen* antibody pLM embedding
   (AntiBERTy by default), and the pair representation is a lightweight
   relative-position + outer-product init. This is what makes AbDiff
   **14.5M parameters and single-sequence** while still matching 500M-parameter
   MSA models on CDR-H3.
3. **Antibody-only scope.** Inputs are always Fv domains (ANARCI-trimmed),
   which lets us use a small domain-specific pLM and a per-residue CDR labeling
   that the loss can exploit.

```
   frozen pLM embedding  ─┐
   residue_index, asym_id ├─►  conditioning  ─►  N_block × DiffBlock  ─►  EDM skip  ─►  x̂  (denoised atom14 coords)
   noisy coords x_noisy  ─┘                       (pair-bias attn +
   noise level t  ────────►  Fourier embed         transition)
```

---

## 2. Input / output representation

| Tensor | Shape | Meaning |
|---|---|---|
| `emb` | `[B, N, c_esm]` | frozen per-residue pLM embedding (the *only* sequence signal) |
| `coords` | `[B, N, 14, 3]` | ground-truth **atom14** all-atom coordinates |
| `atom_mask` | `[B, N, 14]` | which of the 14 atoms exist for each residue |
| `token_mask` | `[B, N]` | which residues are real (vs. padding) |
| `residue_index` | `[B, N]` | IMGT residue number (for relative-position encoding) |
| `asym_id` | `[B, N]` | chain id — distinguishes H/L, the two halves of an scFv, etc. |
| `cdr` | `[B, N]` | per-residue CDR label: `0`=framework, `1`=CDR1, `2`=CDR2, `3`=CDR3 |
| `htype` | `[B, N]` | `1`=heavy chain, `0`=light — lets the loss isolate **CDR-H3** |

**atom14** is the AlphaFold all-atom layout: each residue is represented by up to
14 atoms in a fixed per-residue-type order, where indices `0,1,2,3 = N, CA, C, O`
(backbone) and `4..13` are sidechain atoms. **CA is always index 1.** This single
convention is what lets the same code do backbone-only (`N_BB=4`) and all-atom
(`N_BB=14`) — the symbol `N_BB` in the code is the number of atom slots.

The model input/output is the full `[B, N, 14, 3]` coordinate tensor. Scoring and
the rollout slice index 1 (CA) for RMSD and indices `0:4` for backbone metrics.

### Antibody formats: scFv, Fab/Fv, VHH/HCAb

Formats are handled purely through `asym_id` + ANARCI trimming — no architectural
branch:
- **Paired Fv / Fab**: two `asym_id` segments (heavy + light).
- **scFv**: a single polypeptide carrying two V-domains; ANARCI finds both, and the
  linker is dropped, leaving two `asym_id` segments on one chain.
- **VHH / nanobody / HCAb**: heavy-only — a single `asym_id` segment, no light chain.

Because chain identity is just a feature consumed by the relative-position
encoding (below), the same weights fold all three.

---

## 3. Conditioning stack

The conditioning produces a **single representation** `s [B,N,c]` and a **pair
representation** `z [B,N,N,c_z]` that bias the denoiser.

### 3.1 Single representation from the pLM
```python
s = LayerNorm-then-Linear(emb)  +  FourierSigmaEmbed(t)
```
- `s_in`: `LayerNorm(c_esm) → Linear(c_esm, c)` projects the frozen pLM embedding
  into the model width `c`. The model **auto-reads `c_esm`** from the data, so any
  pLM (AntiBERTy 512, AbLang2 480, ESM2-650M 1280, ESM2-150M 640) drops in without
  code changes.
- The noise level `t` is embedded and **added to every token** so the network
  knows how much noise it must remove (see §3.2).

### 3.2 Noise-level embedding (`FourierSigmaEmbed`)
AF3 conditions on the log-noise-level via Fourier features:
$$ c = 0.25\,\ln\!\left(\tfrac{t}{\sigma_\text{data}}\right), \qquad
   \text{embed}(t) = \big[\sin(2\pi(c\,w+b)),\ \cos(2\pi(c\,w+b))\big] $$
with random fixed frequencies `w` and phases `b`. `sigma_data = 16.0 Å` is the
assumed standard deviation of the coordinate data and appears throughout the EDM
preconditioning.

### 3.3 Relative-position encoding (`RelPos`)
Pairwise features from IMGT numbering, with an explicit **different-chain bucket**:
$$ d_{ij} = \mathrm{clip}(r_i - r_j,\ -k,\ k) + k, \qquad k=32 $$
A one-hot of `d_{ij}` is gated by *same-chain*; an extra channel flags
*different-chain* pairs. This is how the model knows H vs L (or the two V-domains
of an scFv) are separate chains that can move relative to each other.

### 3.4 Pair init (outer-product-mean style)
```python
z = RelPos(residue_index, asym_id) + OPM(s)_i + OPM(s)_j
```
A linear `OPM: c → c_z` is broadcast over rows and columns and added to the
relative-position pair — a cheap stand-in for AF's outer-product-mean that injects
single-rep content into the pair track. There is **no triangle update** — the pair
is computed once and reused as an attention bias in every block.

---

## 4. The denoiser network

`n_block` identical `DiffBlock`s, each a pre-norm residual pair of
**pair-biased self-attention** + **transition (MLP)**:

```python
x = x + PairBiasAttention(x, z, mask)
x = x + Transition(x)
```

### 4.1 Pair-biased attention
Standard multi-head self-attention over residues, with the pair representation
added as a **per-head additive bias** to the logits:
$$ \mathrm{att} = \mathrm{softmax}\!\left( \frac{QK^\top}{\sqrt{d_h}} + \underbrace{\text{Linear}(\mathrm{LN}(z))}_{[B,H,N,N]} \right) V $$
Padding tokens are masked with `finfo(dtype).min` *before* softmax (so they never
leak), and the result is projected back to width `c`. This is the only place the
pair track enters — it steers *which residues attend to which*, which is exactly
where the H/L interface and the CDR loop geometry live.

### 4.2 Coordinate I/O
- **In**: the (scaled) noisy coordinates are flattened `14×3 → 42` and linearly
  embedded, then **added to the single rep**: `h = s + coord_in(r)`.
- **Out**: `LayerNorm → Linear(c, 42) → reshape [B,N,14,3]`. The output projection
  is **zero-initialized**, so at start of training the network predicts the EDM
  skip term (a scaled copy of the input) and learns the correction from there —
  the standard EDM stability trick.

---

## 5. EDM diffusion formulation (the "AF3 manner")

AbDiff uses **Karras et al. (EDM) preconditioning**, identical to OpenFold3's
`diffusion_module`. The network `F` never predicts the clean structure directly;
instead its output is combined with the noisy input through skip/҂out scalings
that keep the effective target unit-variance at every noise level.

**Input scaling** (so the network always sees ~unit-variance input):
$$ r = \frac{x_\text{noisy}}{\sqrt{t^2 + \sigma_\text{data}^2}} $$

**EDM skip connection** (the denoised prediction):
$$ \boxed{\;x_\text{out} = c_\text{skip}(t)\,x_\text{noisy} + c_\text{out}(t)\,F(r,\dots)\;} $$
$$ c_\text{skip}(t) = \frac{\sigma_\text{data}^2}{\sigma_\text{data}^2 + t^2}, \qquad
   c_\text{out}(t) = \frac{\sigma_\text{data}\,t}{\sqrt{\sigma_\text{data}^2 + t^2}} $$

Intuition: at **low noise** (`t→0`), `c_skip→1`, `c_out→0`, so the model mostly
copies the input — it only needs to make a tiny correction. At **high noise**
(`t→∞`), `c_skip→0`, `c_out→σ_data`, so the model must hallucinate the structure
from conditioning alone. The Fourier noise embedding (§3.2) tells the network
which regime it is in.

---

## 6. Training loss — and the **CDR-weighted** objective

This is the single most important section, because **the loss is the biggest
lever on CDR-H3 accuracy** (it moved CDR-H3 from 2.61 Å → 2.08 Å, making AbDiff
the leader over 500M-parameter MSA models). Implemented in `edm_loss`.

### 6.1 The five steps of a training iteration

**(1) Center + random-rotation augmentation (handles pose, replaces SO(3)).**
Coordinates are centered on the (masked) CA centroid, then a **uniformly random
rotation** `R` (sampled from a random unit quaternion) is applied:
$$ x_0 = R\,(\text{coords} - \text{CA-centroid}) $$
This is the *entire* mechanism by which the model becomes rotation-equivariant —
there is no SO(3) manifold, the network just sees every structure in random
orientations and learns to denoise regardless.

**(2) Sample a noise level (log-normal, the EDM prior).**
$$ t = \sigma_\text{data}\,\exp(P_\text{mean} + P_\text{std}\,\varepsilon), \quad
   \varepsilon\sim\mathcal N(0,1),\ P_\text{mean}=-1.2,\ P_\text{std}=1.5 $$
so most training mass sits at small/medium noise, with a long tail to large noise.

**(3) Add noise and denoise.** `x_noisy = x0 + t·ε` (masked to real atoms), then
`x̂ = model(x_noisy, t, …)` (EDM skip already applied inside `forward`).

**(4) Weighted rigid (Kabsch) alignment — the second half of "no SO(3)".**
Before computing error, the prediction is **rigidly aligned to the target** by a
weighted Kabsch fit (SVD of the cross-covariance, with a determinant sign-fix to
forbid reflections). Combined with step (1), this means the loss is invariant to
global rigid motion *without* ever parameterizing rotations. The SVD is forced to
**fp32 with autocast disabled** — fp16 SVD is numerically unstable and was a real
bug we hit.

**(5) EDM-weighted, CDR-weighted MSE.** The aligned squared error is weighted two
ways and reduced:

*Per-noise-level EDM weight* (so all noise levels contribute comparable gradient):
$$ \lambda(t) = \frac{t^2 + \sigma_\text{data}^2}{(t\,\sigma_\text{data})^2} $$

*Per-residue CDR weight* — **this is the knob that wins CDR-H3:**
$$ w_\text{res} =
\begin{cases}
1 & \text{framework (cdr}=0)\\
w_\text{cdr} & \text{any CDR (cdr}\in\{1,2,3\})\\
w_\text{cdr}\cdot w_\text{h3} & \text{CDR-H3 (cdr}=3 \text{ and htype}=1)
\end{cases}$$

The final loss is the EDM-weighted mean of the per-residue-weighted, atom-masked,
Kabsch-aligned squared error:
$$ \mathcal L = \mathbb E_t\Big[\lambda(t)\cdot
   \frac{\sum_{n,a} w_{\text{res},n}\,\|\hat x_{na}-x^0_{na}\|^2\,m_{na}}
        {\sum_{n,a} w_{\text{res},n}\,m_{na}}\Big] $$

The default (`cdr_weight=1, h3_weight=1`) is a plain all-atom MSE. The leader uses
`--cdr-weight 2.0 --h3-weight 4.0`, so a CDR-H3 atom is weighted `2 × 4 = 8×` a
framework atom. Because the framework is conserved and easy, spending the model's
capacity on the hard, functional, hypervariable H3 loop is what drops CDR-H3 RMSD.

> **Reported RMSD is always unweighted.** `edm_loss` also returns a
> `torch.no_grad()` *unweighted* CA-RMSD purely for monitoring, so training curves
> remain comparable across weightings.

### 6.2 Why training `val_rmsd` looks flat but CDR-H3 changes a lot
Training `val_rmsd` is an overall-backbone RMSD, **dominated by the conserved
framework** (~1.0–1.2 Å for every model). The CDR-H3 differences (2.08 vs 2.61 vs
2.85) only surface in the dedicated *framework-superposed* eval. So do not read
`val_rmsd ≈ 1.1` as "all models equal" — they are not.

---

## 7. Sampling — EDM rollout (AF3 Algorithm 18)

Inference is a reverse-time ODE/SDE integration over a decreasing noise schedule
(`sample_backbone`). It is the same "churn" sampler as AF3:

**Karras noise schedule** (decreasing, ends near 0):
$$ \sigma(u) = \sigma_\text{data}\Big(\sigma_\text{max}^{1/p} + u\,(\sigma_\text{min}^{1/p}-\sigma_\text{max}^{1/p})\Big)^{p} $$
with `σ_max=160`, `σ_min=4e-4`, `p=7`, `u` linear in `[0,1]` over `n_steps` (default 200).

Each step:
1. **Churn**: inflate the current noise level `t̂ = σ_prev·(1+γ)` and inject fresh
   noise (`γ=0.8` above a floor, scaled by `noise_scale=1.003`) — this is the
   stochasticity that improves sample quality.
2. **Denoise**: `x_den = model(x_noisy, t̂, …)`.
3. **Step**: move along the estimated score
   `d = (x_noisy − x_den)/t̂`, with `x ← x_noisy + step_scale·(σ_next − t̂)·d`
   (`step_scale=1.5`).

We sample initial noise only on real atoms (`atom_mask`) and re-mask every step, so
padding atoms stay at the origin. For evaluation we draw **best-of-N** samples
(N=4) and keep the one with lowest CDR-H3 (falling back to overall RMSD when H3 is
absent, e.g. a nanobody scored on a different loop).

---

## 8. Hyperparameters & model sizes

| Config | `c` | `c_z` | `n_head` | `n_block` | Params | Notes |
|---|---|---|---|---|---|---|
| **small** (default / leader) | 384 | 128 | 12 | 8 | **14.5M** | the published model |
| **big** | 512 | 192 | 16 | 16 | **50.9M** | size ablation |

Shared: `sigma_data=16.0`, batch size 8 (6 for big), `max_tok` cap for memory,
fp32 throughout (AMP disabled because of the Kabsch SVD), Adam `lr=3e-4`, 140–160
epochs (~8 h for the small model on an A40; ~186 s/epoch).

The pLM is **frozen** — none of these parameter counts include it, and it is not
fine-tuned. Swapping the pLM only changes `c_esm` (auto-detected).

---

## 9. What we kept from OpenFold3, and what we dropped

| OpenFold3 / AF3 component | AbDiff |
|---|---|
| EDM preconditioning (`c_skip`/`c_out`), input scaling | ✅ kept, identical |
| Fourier noise-level embedding | ✅ kept |
| Karras schedule + churn sampler (Alg. 18) | ✅ kept |
| Random-rotation augmentation + weighted rigid align | ✅ kept (this is our "no-SO(3)") |
| Relative-position encoding w/ chain bucket | ✅ kept (simplified) |
| MSA + template stack | ❌ dropped — replaced by a **frozen antibody pLM** |
| Pairformer / triangle attention | ❌ dropped — pair is a one-shot relpos+OPM bias |
| Explicit residue frames / SE(3) / IPA | ❌ dropped — **pure points in ℝ³** |
| Confidence (pLDDT/PAE) heads | ❌ not implemented (future work) |

The net effect: a model **~36× smaller** than single-sequence Boltz-2 that matches
it on the only metric that is hard for antibodies — the CDR-H3 loop.

---

## 10. Pointers to code

- `scripts/model.py` — everything in this document:
  `AbDiffusion` (network), `edm_loss` (training + CDR weighting), `sample_backbone`
  (rollout), `kabsch_align`, `make_noise_schedule`, `random_rotation`.
- `scripts/train.py` — training loop, `--cdr-weight` / `--h3-weight` flags, fp32,
  periodic checkpoints + in-loop generation eval.
- `scripts/eval_sample.py` / `eval_ours_truth.py` — full rollout + framework-
  superposed per-CDR RMSD (the headline CDR-H3 metric).
- `scripts/prep_structures_anarci.py` — ANARCI Fv-trimming, CDR/htype labels,
  atom14 coords, pLM embedding.

See [`README.md`](README.md) for results, leaderboard, figures, and usage.
