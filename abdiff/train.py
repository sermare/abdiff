#!/usr/bin/env python
"""
AbDiff trainer — AF3-style backbone coordinate diffusion from pLM embeddings.

Runs an eval-before-train sanity pass (shapes, finiteness, one rollout), then trains.
"""
import argparse, glob, json, os, time
import torch
from torch.utils.data import Dataset, DataLoader
from model import AbDiffusion, edm_loss, sample_backbone, kabsch_align, N_BB


def log(*a): print(*a, flush=True)


@torch.no_grad()
def generation_eval(model, files, dev, n, steps):
    """Honest metric: full diffusion rollout from noise -> Kabsch-aligned CA-RMSD."""
    model.eval(); rmsds = []
    for fp in files[:n]:
        rec = torch.load(fp, map_location="cpu")
        b = to_dev(collate([rec]), dev)
        gt = b["coords"][0]; m = b["atom_mask"][0]
        x = sample_backbone(model, b["emb"], b["residue_index"], b["asym_id"],
                            b["token_mask"], b["atom_mask"], n_steps=steps)[0]
        Pa = kabsch_align(x.reshape(1, -1, 3), gt.reshape(1, -1, 3),
                          m.reshape(1, -1).float()).reshape(-1, N_BB, 3)
        cam = m[:, 1]
        if cam.any():
            rmsds.append(torch.sqrt(((Pa[:, 1] - gt[:, 1])[cam] ** 2).sum(-1).mean()).item())
    model.train()
    return sum(rmsds) / max(len(rmsds), 1)


class AbSet(Dataset):
    def __init__(self, files, max_tok=512):
        self.files = files; self.max_tok = max_tok
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        r = torch.load(self.files[i], map_location="cpu")
        n = r["emb"].shape[0]
        if n > self.max_tok:
            r = {k: (v[:self.max_tok] if torch.is_tensor(v) and v.shape and v.shape[0] == n else v)
                 for k, v in r.items()}
        return r


def collate(items):
    N = max(r["emb"].shape[0] for r in items)
    B = len(items)
    c_esm = items[0]["emb"].shape[1]
    emb = torch.zeros(B, N, c_esm)
    coords = torch.zeros(B, N, N_BB, 3)
    amask = torch.zeros(B, N, N_BB, dtype=torch.bool)
    asym = torch.zeros(B, N, dtype=torch.long)
    residx = torch.zeros(B, N, dtype=torch.long)
    tmask = torch.zeros(B, N)
    cdr = torch.zeros(B, N, dtype=torch.long)
    htype = torch.zeros(B, N, dtype=torch.long)
    for b, r in enumerate(items):
        n = r["emb"].shape[0]
        emb[b, :n] = r["emb"].float()
        coords[b, :n] = r["coords"]
        amask[b, :n] = r["atom_mask"]
        asym[b, :n] = r["asym_id"].long()
        residx[b, :n] = r["residue_index"].long()
        tmask[b, :n] = 1.0
        if "cdr" in r:
            cdr[b, :n] = r["cdr"].long(); htype[b, :n] = r["htype"].long()
    return {"emb": emb, "coords": coords, "atom_mask": amask, "cdr": cdr, "htype": htype,
            "asym_id": asym, "residue_index": residx, "token_mask": tmask}


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def sanity(model, batch, dev):
    log("[sanity] eval-before-train")
    model.eval()
    with torch.no_grad():
        b = to_dev(batch, dev)
        x = b["coords"]
        t = torch.full((x.shape[0],), model.sigma_data, device=dev)
        out = model(x + t[:, None, None, None] * torch.randn_like(x), t,
                    b["emb"], b["residue_index"], b["asym_id"], b["token_mask"])
        loss, rmsd = edm_loss(model, b)
        params = sum(p.numel() for p in model.parameters())
        log(f"[sanity] params={params/1e6:.2f}M  in={tuple(x.shape)} out={tuple(out.shape)}")
        log(f"[sanity] out finite={torch.isfinite(out).all().item()}  loss={loss.item():.4f}  rmsd={rmsd.item():.3f}")
        assert out.shape == x.shape and torch.isfinite(out).all() and torch.isfinite(loss)
    log("[sanity] PASS")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/train")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c", type=int, default=384)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-block", type=int, default=8)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--save-every", type=int, default=20)    # periodic epoch checkpoints
    ap.add_argument("--eval-every", type=int, default=20)    # in-training generation eval
    ap.add_argument("--gen-eval-n", type=int, default=10)    # #held-out Fvs per gen eval
    ap.add_argument("--gen-eval-steps", type=int, default=100)
    ap.add_argument("--resume", default="")                  # ckpt to warm-start from
    ap.add_argument("--cdr-weight", type=float, default=1.0)  # upweight all CDR residues in loss
    ap.add_argument("--h3-weight", type=float, default=1.0)   # extra multiplier on CDR-H3
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    assert files, f"no training tensors in {args.data}"
    n_val = max(1, len(files) // 10)
    train_f, val_f = files[n_val:], files[:n_val]
    log(f"[data] train={len(train_f)} val={len(val_f)} device={dev}")

    c_esm = torch.load(files[0], map_location="cpu")["emb"].shape[1]
    model = AbDiffusion(c_esm=c_esm, c=args.c, c_z=args.c_z, n_head=args.n_head,
                        n_block=args.n_block).to(dev)
    log(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
        f"(c={args.c} c_z={args.c_z} heads={args.n_head} blocks={args.n_block})")
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev)["model"])
        log(f"[resume] warm-started from {args.resume}")

    drop_last = len(train_f) >= 2 * args.bs   # avoid emptying tiny datasets
    dl = DataLoader(AbSet(train_f, args.max_tok), batch_size=args.bs, shuffle=True,
                    collate_fn=collate, num_workers=2, drop_last=drop_last)
    vdl = DataLoader(AbSet(val_f, args.max_tok), batch_size=args.bs, shuffle=False,
                     collate_fn=collate, num_workers=1)

    sanity_batch = collate([AbSet(train_f, args.max_tok)[i]
                            for i in range(min(args.bs, len(train_f)))])
    sanity(model, sanity_batch, dev)
    if args.sanity_only:
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # fp32 throughout: model is small (~14M params, N<=320) so it fits on the A40,
    # and this avoids the whole class of fp16/autocast issues (svd, masked_fill, ...).
    best = 1e9
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for batch in dl:
            b = to_dev(batch, dev)
            opt.zero_grad()
            loss, rmsd = edm_loss(model, b, cdr_weight=args.cdr_weight, h3_weight=args.h3_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        # val
        model.eval(); vtot = vr = vn = 0.0
        with torch.no_grad():
            for batch in vdl:
                b = to_dev(batch, dev)
                l, r = edm_loss(model, b)
                vtot += l.item(); vr += r.item(); vn += 1
        vl = vtot / max(vn, 1); vrm = vr / max(vn, 1)
        log(f"[ep {ep:03d}] train={tot/max(nb,1):.4f} val={vl:.4f} val_rmsd={vrm:.3f} "
            f"({time.time()-t0:.1f}s)")
        snap = {"model": model.state_dict(), "args": vars(args),
                "c_esm": c_esm, "epoch": ep, "val": vl}
        torch.save(snap, os.path.join(args.ckpt_dir, "abdiff_last.pt"))  # resilience
        if vl < best:
            best = vl
            torch.save(snap, os.path.join(args.ckpt_dir, "abdiff_best.pt"))
        # periodic checkpoint + honest generation eval at a few points
        is_last = (ep == args.epochs - 1)
        if args.save_every and (ep % args.save_every == 0 or is_last):
            torch.save(snap, os.path.join(args.ckpt_dir, f"abdiff_ep{ep:03d}.pt"))
        if args.eval_every and (ep % args.eval_every == 0 or is_last) and ep > 0:
            g = generation_eval(model, val_f, dev, args.gen_eval_n, args.gen_eval_steps)
            log(f"[gen-eval ep {ep:03d}] rollout CA-RMSD = {g:.3f} A "
                f"(n={args.gen_eval_n}, steps={args.gen_eval_steps})")
    log(f"[done] best_val={best:.4f}")


if __name__ == "__main__":
    main()
