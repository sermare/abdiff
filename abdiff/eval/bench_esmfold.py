#!/usr/bin/env python
"""
Stage 2 (struct-evo env, has ESMFold): fold the held-out Fv sequences with ESMFold
(single-sequence, no MSA) and score CDR-H3 etc. with the SAME framework-superpose
protocol as AbDiff's eval. Reads data/bench_truth.pt (from bench_prep_truth.py).
Self-contained: plain PDB parsing + inline Kabsch (no biotite/anarci needed here).
"""
import argparse, os
import numpy as np
import torch

WEIGHTS = os.environ.get("ESMFOLD_WEIGHTS", "")
BB = ["N", "CA", "C", "O"]


def log(*a): print(*a, flush=True)


def parse_pdb_backbone(pdb_str):
    """Return [Nres,4,3] backbone coords in residue order from an ESMFold PDB string."""
    res, cur, curkey = [], None, None
    for ln in pdb_str.splitlines():
        if not ln.startswith("ATOM"):
            continue
        an = ln[12:16].strip(); key = (ln[21], ln[22:27])
        if key != curkey:
            if cur is not None:
                res.append(cur)
            cur = np.zeros((4, 3), np.float32); curkey = key
        if an in BB:
            i = BB.index(an)
            cur[i] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    if cur is not None:
        res.append(cur)
    return np.stack(res) if res else np.zeros((0, 4, 3), np.float32)


def kabsch(P, Q, w):
    w = w[:, None]
    Pc = (P * w).sum(0) / w.sum().clamp_min(1e-6)
    Qc = (Q * w).sum(0) / w.sum().clamp_min(1e-6)
    P0, Q0 = P - Pc, Q - Qc
    H = (w * P0).T @ Q0
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.T @ U.T))
    D = torch.eye(3); D[2, 2] = d
    R = Vh.T @ D @ U.T
    return (P0 @ R.T) + Qc


def ca_rmsd(P, Q, m):
    if m.sum() == 0:
        return float("nan")
    return torch.sqrt(((P[:, 1] - Q[:, 1]) ** 2).sum(-1)[m].mean()).item()


def score(pred, gt, mask, cdr, htype):
    fr = (cdr == 0)
    w = (mask & fr[:, None]).reshape(-1).float()
    if w.sum() < 12:
        w = mask.reshape(-1).float()
    Pa = kabsch(pred.reshape(-1, 3), gt.reshape(-1, 3), w).reshape(-1, 4, 3)
    cam = mask[:, 1]
    return {"all": ca_rmsd(Pa, gt, cam), "fr": ca_rmsd(Pa, gt, cam & fr),
            "cdr1": ca_rmsd(Pa, gt, cam & (cdr == 1)), "cdr2": ca_rmsd(Pa, gt, cam & (cdr == 2)),
            "cdr3": ca_rmsd(Pa, gt, cam & (cdr == 3)),
            "h3": ca_rmsd(Pa, gt, cam & (cdr == 3) & (htype == 1)),
            "l3": ca_rmsd(Pa, gt, cam & (cdr == 3) & (htype == 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="./data/bench_truth.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.hub.set_dir(WEIGHTS)
    import esm
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    log(f"[esmfold] loading model (device={dev}) ...")
    model = esm.pretrained.esmfold_v1().eval().to(dev)
    model.set_chunk_size(128)

    truth = torch.load(args.truth, map_location="cpu")
    agg = {}
    for hid, t in truth.items():
        try:
            with torch.no_grad():
                pdb = model.infer_pdb(":".join(t["seqs"]))
            pred = parse_pdb_backbone(pdb)
            gt = torch.tensor(t["coords"]); m = torch.tensor(t["mask"])
            if pred.shape[0] != gt.shape[0]:
                log(f"  [skip] {hid} len pred={pred.shape[0]} gt={gt.shape[0]}"); continue
            gt = gt - gt[:, 1][m[:, 1]].mean(0)[None, None, :]
            r = score(torch.tensor(pred), gt, m, torch.tensor(t["cdr"]).long(),
                      torch.tensor(t["htype"]).long())
        except Exception as e:
            log(f"  [skip] {hid}: {e}"); continue
        f = t["format"]; d = agg.setdefault(f, {})
        for k, v in r.items():
            if v == v:
                d.setdefault(k, []).append(v)
        log(f"  {hid:13s} {f:4s} N={gt.shape[0]:4d} all={r['all']:.2f} fr={r['fr']:.2f} "
            f"H3={r['h3'] if r['h3']==r['h3'] else float('nan'):.2f}")

    log("\n[ESMFold] mean RMSD (Å), framework-superposed:")
    log(f"  {'fmt':5s} {'n':>4s} {'all':>6s} {'fr':>6s} {'cdr1':>6s} {'cdr2':>6s} {'cdr3':>6s} {'H3':>6s} {'L3':>6s}")
    for f, d in sorted(agg.items()):
        n = len(d.get("all", []))
        def mu(k): return (sum(d[k]) / len(d[k])) if d.get(k) else float("nan")
        log(f"  {f:5s} {n:4d} {mu('all'):6.2f} {mu('fr'):6.2f} {mu('cdr1'):6.2f} "
            f"{mu('cdr2'):6.2f} {mu('cdr3'):6.2f} {mu('h3'):6.2f} {mu('l3'):6.2f}")
    allh3 = [v for d in agg.values() for v in d.get("h3", [])]
    if allh3:
        log(f"[ESMFold] *** overall CDR-H3 Cα-RMSD = {sum(allh3)/len(allh3):.2f} Å (n={len(allh3)}) ***")


if __name__ == "__main__":
    main()
