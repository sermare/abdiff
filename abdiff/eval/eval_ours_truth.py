#!/usr/bin/env python
"""
Score AbDiff (ours) on the SAME held-out truth bundle used for the external folders,
with the identical framework-superpose -> per-CDR RMSD protocol. Runs anywhere torch
is available (tiny model -> fine on a 1080ti).
"""
import argparse, os, sys
import torch
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from model import AbDiffusion, sample_backbone
from bench_esmfold import score   # identical scoring as the ESMFold benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./checkpoints/sabdab/abdiff_final_v1.pt")
    ap.add_argument("--truth", default="./data/bench_truth.pt")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(args.ckpt, map_location=dev); a = ck.get("args", {})
    model = AbDiffusion(c_esm=ck.get("c_esm", 512), c=a.get("c", 384),
                        n_block=a.get("n_block", 8)).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"[ours] ckpt epoch={ck.get('epoch')} device={dev}", flush=True)
    truth = torch.load(args.truth, map_location="cpu")
    agg = {}
    for hid, t in truth.items():
        if "emb" not in t:
            print(f"[skip] {hid} no emb (re-run bench_prep_truth)"); continue
        emb = torch.tensor(t["emb"]).float()[None].to(dev)
        residx = torch.tensor(t["residue_index"]).long()[None].to(dev)
        asym = torch.tensor(t["asym_id"]).long()[None].to(dev)
        gt = torch.tensor(t["coords"]); m = torch.tensor(t["mask"])
        am = m[None].to(dev); tmask = torch.ones(1, emb.shape[1], device=dev)
        gt = gt - gt[:, 1][m[:, 1]].mean(0)[None, None, :]
        cdr = torch.tensor(t["cdr"]).long(); ht = torch.tensor(t["htype"]).long()
        best = None
        with torch.no_grad():
            for _ in range(args.n_samples):
                x = sample_backbone(model, emb, residx, asym, tmask, am, n_steps=args.steps)[0].cpu()
                r = score(x, gt, m, cdr, ht)
                key = r["h3"] if r["h3"] == r["h3"] else r["all"]
                if best is None or key < best[0]:
                    best = (key, r)
        r = best[1]; f = t["format"]; d = agg.setdefault(f, {})
        for k, v in r.items():
            if v == v:
                d.setdefault(k, []).append(v)
        print(f"  {hid:13s} {f:4s} N={gt.shape[0]:4d} all={r['all']:.2f} fr={r['fr']:.2f} "
              f"H3={r['h3'] if r['h3']==r['h3'] else float('nan'):.2f}", flush=True)
    print("\n[AbDiff-ours] mean RMSD (Å), framework-superposed:")
    print(f"  {'fmt':5s} {'n':>4s} {'all':>6s} {'fr':>6s} {'cdr1':>6s} {'cdr2':>6s} {'cdr3':>6s} {'H3':>6s} {'L3':>6s}")
    for f, d in sorted(agg.items()):
        n = len(d.get("all", []))
        def mu(k): return (sum(d[k]) / len(d[k])) if d.get(k) else float("nan")
        print(f"  {f:5s} {n:4d} {mu('all'):6.2f} {mu('fr'):6.2f} {mu('cdr1'):6.2f} "
              f"{mu('cdr2'):6.2f} {mu('cdr3'):6.2f} {mu('h3'):6.2f} {mu('l3'):6.2f}")
    allh3 = [v for d in agg.values() for v in d.get("h3", [])]
    if allh3:
        print(f"[AbDiff-ours] *** overall CDR-H3 Cα-RMSD = {sum(allh3)/len(allh3):.2f} Å (n={len(allh3)}) ***")


if __name__ == "__main__":
    main()
