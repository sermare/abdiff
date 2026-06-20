#!/usr/bin/env python
"""
Evaluate a SERIES of saved checkpoints (abdiff_ep*.pt) with the full diffusion
rollout, to chart generation CA-RMSD vs training epoch. Writes a CSV + prints a table.
"""
import argparse, glob, os, re
import torch
from model import AbDiffusion, sample_backbone, kabsch_align, N_BB
from train import AbSet, collate, to_dev


@torch.no_grad()
def eval_ckpt(path, val_files, dev, n, samples, steps):
    ck = torch.load(path, map_location=dev)
    a = ck.get("args", {})
    model = AbDiffusion(c_esm=ck.get("c_esm", 512), c=a.get("c", 384), c_z=a.get("c_z", 128), n_head=a.get("n_head", 12), n_block=a.get("n_block", 8)).to(dev).eval()
    model.load_state_dict(ck["model"])
    by_fmt, allv = {}, []
    for fp in val_files[:n]:
        rec = torch.load(fp, map_location="cpu")
        b = to_dev(collate([rec]), dev)
        gt = b["coords"][0]; m = b["atom_mask"][0]
        best = None
        for _ in range(samples):
            x = sample_backbone(model, b["emb"], b["residue_index"], b["asym_id"],
                                b["token_mask"], b["atom_mask"], n_steps=steps)[0]
            Pa = kabsch_align(x.reshape(1, -1, 3), gt.reshape(1, -1, 3),
                              m.reshape(1, -1).float()).reshape(-1, N_BB, 3)
            cam = m[:, 1]
            r = torch.sqrt(((Pa[:, 1] - gt[:, 1])[cam] ** 2).sum(-1).mean()).item()
            best = r if best is None else min(best, r)
        by_fmt.setdefault(rec["format"], []).append(best); allv.append(best)
    mean = sum(allv) / max(len(allv), 1)
    fmt_means = {f: sum(v) / len(v) for f, v in by_fmt.items()}
    return ck.get("epoch", -1), ck.get("val", float("nan")), mean, fmt_means, len(allv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="./checkpoints/sabdab")
    ap.add_argument("--data", default="./data/train_sabdab")
    ap.add_argument("--glob", default="abdiff_ep*.pt")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-csv", default="./results/ckpt_curve.csv")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    val_files = files[:max(1, len(files) // 10)]
    cks = sorted(glob.glob(os.path.join(args.ckpt_dir, args.glob)),
                 key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    print(f"[ckpt-eval] {len(cks)} checkpoints, {len(val_files)} held-out, device={dev}")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    for p in cks:
        ep, val, mean, fmt_means, n = eval_ckpt(p, val_files, dev, args.n_eval,
                                                 args.n_samples, args.steps)
        fmt_str = " ".join(f"{f}={v:.2f}" for f, v in sorted(fmt_means.items()))
        print(f"  ep{ep:03d}  val={val:.4f}  gen_CA_RMSD={mean:.2f} A  [{fmt_str}]  (n={n})")
        rows.append((ep, val, mean, fmt_means))
    with open(args.out_csv, "w") as f:
        f.write("epoch,val_loss,gen_ca_rmsd,fab,vhh,scfv\n")
        for ep, val, mean, fm in rows:
            f.write(f"{ep},{val:.4f},{mean:.3f},{fm.get('fab','')},{fm.get('vhh','')},{fm.get('scfv','')}\n")
    print(f"[ckpt-eval] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
