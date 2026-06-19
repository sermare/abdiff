# AbDiff — an AF3-style antibody structure model from pLM embeddings

**Goal.** Predict antibody 3D structure (all-atom, R³ coordinate diffusion — *no* SO(3)/frame
manifold) directly from protein-language-model (pLM) residue embeddings, by transplanting
OpenFold3's diffusion module and replacing its MSA+template+Evoformer trunk with a thin
pLM-conditioned trunk. Must natively cover **paired Fab/Fv, scFv, and heavy-chain-only
antibodies (HCAb / VHH / nanobody)**.

---

## 1. Why this is tractable (resources on disk)

| Piece | Location | Notes |
|---|---|---|
| OpenFold3 (AF3 reproduction) | `/global/scratch/users/sergiomar10/AF3/openfold-3/` | full diffusion stack |
| AF3 coordinate-diffusion module | `openfold3/core/model/structure/diffusion_module.py` | `DiffusionModule` (Alg 20), `SampleDiffusion` (Alg 18) |
| Diffusion building blocks | `openfold3/core/model/layers/{diffusion_conditioning,diffusion_transformer,sequence_local_atom_attention,attention_pair_bias}.py` | |
| Trunk we will mostly drop | `openfold3/core/model/latent/{pairformer,evoformer,msa_module,template_module}.py` | keep a *few* Pairformer blocks |
| Feature embedders | `openfold3/core/model/feature_embedders/input_embedders.py` | ref-conformer / atom featurization reused |
| pLM weights (offline) | `loop/hf_cache/hub/models--facebook--esm2_*` | ESM2 8M/35M/150M/650M/3B, esm1v, esm3-sm, ProtBert |
| Antibody seqs (SAbDab-derived) | `loop/data/antibody_only_manifest*.json` | `chain_map` H/L(/M/K) sequences per PDB |
| SAbDab structures (labels) | `loop/results/{sprint31_sabdab,pilot_sabdab}` + (locate Chothia PDBs) | **TODO: confirm coord files for training targets** |
| Env | conda `tttppi` @ `/clusterfs/nilah/sergio/miniconda3/envs/tttppi` (torch 2.1.0+cu118) | |
| Compute | SLURM `savio3_gpu`, qos `savio_lowprio`, acct `co_nilah`, `gpu:A40:1` | **login-node python over NFS is unusably slow → all compute via sbatch** |

The OF3 diffusion module is **already** EDM-style coordinate diffusion in R³ (see
`DiffusionModule.forward`: noisy atoms → denoised atoms, EDM preconditioning, no frames).
`centre_random_augmentation` applies a *random global* rotation+translation as data
augmentation only — it is **not** SO(3)-manifold diffusion. This is exactly the "no SO3
space" the user asked for; we keep it as-is.

---

## 2. Architecture

```
            per-chain pLM (ESM2)                     reference-conformer features
  seq(s) ───────────────────────► s_plm[N_tok, c_esm]   (idealized aa atoms, element, atom→token map)
                                       │                              │
                          Linear+LN ───┤                              │
                                       ▼                              │
                              s_input, s_single[N_tok, c_s]           │
                                       │                              │
        relpos(asym_id, residue_index, entity_id)                    │
                                       ▼                              │
                          thin Pairformer (K≈4 blocks)                │
                                       ▼                              │
                              z_pair[N_tok, N_tok, c_z]               │
                                       │                              │
                                       ▼                              ▼
        ┌────────────────────  DiffusionModule (OF3, unchanged)  ─────────────────┐
        │  DiffusionConditioning(s_input, s_single, z_pair, t)                    │
        │  AtomAttentionEncoder(noisy atoms, ref feats, z) → atom/token acts      │
        │  DiffusionTransformer(token acts | s, z)                                │
        │  AtomAttentionDecoder → per-atom position update                        │
        └─────────────────────────────────────────────────────────────────────► x̂ atoms (R³)
```

What we **reuse verbatim** from OF3: `DiffusionModule`, `SampleDiffusion`, `DiffusionConditioning`,
`DiffusionTransformer`, atom encoder/decoder, the EDM noise schedule, and the protein
reference-conformer featurization.

What we **replace**: the MSA module, template module, and the deep Evoformer. Instead the
single representation comes from a **frozen pLM** (ESM2-150M to start, per the project note;
upgradeable to 650M), and the pair representation comes from a **shallow Pairformer** seeded by
relative-position encoding + outer-product of the single rep. No MSA is ever built — appropriate
for antibodies, whose diversity is in the CDRs, not in evolutionary couplings.

Dimensions (initial): `c_esm`=640 (150M) / 1280 (650M), `c_s`=384, `c_z`=128, Pairformer
blocks K=4, diffusion module at OF3 defaults. All tunable in `configs/`.

---

## 3. The key question: Fab vs scFv vs HCAb/VHH — one representation, no architectural forks

AF3/OF3 already represent *arbitrary multi-chain complexes* purely through **per-token features**;
antibody format is just a particular setting of those features. The atom-diffusion stack never
needs to know "this is a Fab" vs "this is a nanobody".

Per-token features that encode format:
- `residue_index` — position **within a chain** (resets per chain)
- `asym_id` — chain *instance* id (0,1,2,…)
- `entity_id` / `sym_id` — for identical copies
- relative-position encoding uses these: same-chain offsets get a clamped bucket; cross-chain
  pairs get a dedicated "different chain" bucket.

pLM rule: **embed each polypeptide chain independently** (ESM2 is single-sequence), then
concatenate the per-residue embeddings token-wise in chain order. Cross-chain geometry is
recovered downstream by the Pairformer + the diffusion module's pair bias.

| Format | #polypeptide chains | pLM calls | asym_id | residue_index | Notes |
|---|---|---|---|---|---|
| **Fab / Fv (paired VH+VL)** | 2 | embed H, embed L | H=0, L=1 | resets H→L | relpos puts H↔L in the cross-chain bucket |
| **scFv (VH–linker–VL)** | 1 | embed the whole single chain (linker incl.) | all = 0 | continuous through VH+linker+VL | physically *is* one chain; pLM sees the GS-linker → learns the tether. Optional extra `domain_id` feature (VH=0/VL=1) but not required |
| **HCAb / VHH / nanobody** | 1 | embed the single VH(H) domain (~120 aa) | all = 0 | continuous | degenerate single-chain case; trivially handled |
| **+ antigen (later)** | +1 per chain | embed antigen chain(s) | next ids | resets | same mechanism extends to Ab–antigen complexes |

Consequences:
- **No model code branches per format.** A data adapter emits `(tokens, asym_id, residue_index,
  entity_id, atom features, coords)`; everything else is shared.
- scFv linkers (typically (G4S)×3) are kept as real residues so the pLM and geometry see the
  covalent connectivity; we just don't supervise/score linker coordinates heavily (down-weight
  in loss, since linkers are flexible/often disordered).
- VHH framework-2 hallmark residues differ from paired VH; the pLM already encodes this, so no
  special casing — but we'll keep a `is_vhh` tag for evaluation slicing.
- Optional upgrade: a **paired** antibody LM (IgBert/AntiBERTy) or ESM3 chain-break token to give
  the single rep cross-chain context for Fabs. Baseline = per-chain ESM2; the Pairformer
  compensates. Not on disk yet — flagged as a stretch.

---

## 4. Build order (we are at step 1–2)

1. **Sequences** — parse SAbDab manifest → `data/ab_corpus.jsonl`, each record:
   `{id, format∈{fab,scfv,vhh}, chains:[{name,seq,asym_id,domain}], …}`. Classify by chain
   composition; detect scFv by a single chain that contains *two* V-domains (linker search).
2. **pLM embeddings** — `embed_plm.py`: per-chain ESM2-150M residue embeddings, cached to
   `data/emb/<id>.pt`. (sbatch, GPU.)  ← *"start with the plm"*
3. **Structure labels** — locate/standardize SAbDab coordinates (Chothia/IMGT), build per-token
   atom arrays + masks aligned to the sequence tokens. (Next data task.)
4. **Trunk adapter** — pLM→`s`, relpos+OPM→Pairformer→`z`; wire into `DiffusionModule`.
5. **Eval-before-train** — instantiate the full forward pass on a toy batch, run
   `SampleDiffusion` rollout, check shapes/finiteness/equivariance-of-augmentation, and a
   single-example overfit sanity test. **Then** launch training.

---

## 5. Open items
- Confirm SAbDab coordinate files (training targets) and renumbering scheme on disk.
- Decide tokenization: residue-level tokens (AF3 standard) with all-atom reference conformers.
- Loss: AF3 diffusion loss (weighted MSE on aligned atoms) + optional smooth-LDDT; down-weight
  scFv linkers.
- Whether to fine-tune the pLM (LoRA) later vs. frozen embeddings first (start frozen).
