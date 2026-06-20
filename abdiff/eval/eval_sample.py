#!/usr/bin/env python
"""
Evaluate an AbDiff checkpoint with the full diffusion rollout, reporting per-CDR
backbone RMSD using the standard antibody protocol: superpose on the FRAMEWORK,
then measure loop RMSD in that frame. CDR-H3 is the headline metric.

Requires tensors prepped with per-token 'cdr' (0=FR,1,2,3) and 'htype' (1=heavy) fields.
Embeddings are precomputed, so this runs fine on CPU for a few examples.
"""
import argparse, glob, os
import torch
from model import AbDiffusion, sample_backbone, kabsch_align, N_BB
from train import AbSet, collate, to_dev

BB_NAMES = ["N", "CA", "C", "O"]


def ca_rmsd(pred, gt, res_mask):
    """Cα-RMSD over residues where res_mask is True (pred/gt already aligned). [N,4,3]"""
    if res_mask.sum() == 0:
        return float("nan")
    d = ((pred[:, 1] - gt[:, 1]) ** 2).sum(-1)[res_mask]
    return torch.sqrt(d.mean()).item()


def region_rmsds(pred, gt, atom_mask, cdr, htype):
    """Framework-superpose, then per-region Cα-RMSD. Returns dict + aligned pred."""
    fr = (cdr == 0)                                  # framework residues
    # framework backbone atoms as alignment weights
    w = (atom_mask & fr[:, None]).reshape(1, -1).float()
    if w.sum() < 12:                                  # too little framework -> global align
        w = atom_mask.reshape(1, -1).float()
    Pa = kabsch_align(pred.reshape(1, -1, 3), gt.reshape(1, -1, 3), w).reshape(-1, N_BB, 3)
    cam = atom_mask[:, 1]
    out = {
        "all":  ca_rmsd(Pa, gt, cam),
        "fr":   ca_rmsd(Pa, gt, cam & fr),
        "cdr1": ca_rmsd(Pa, gt, cam & (cdr == 1)),
        "cdr2": ca_rmsd(Pa, gt, cam & (cdr == 2)),
        "cdr3": ca_rmsd(Pa, gt, cam & (cdr == 3)),
        "h3":   ca_rmsd(Pa, gt, cam & (cdr == 3) & (htype == 1)),
        "l3":   ca_rmsd(Pa, gt, cam & (cdr == 3) & (htype == 0)),
    }
    return out, Pa


AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
       "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
       "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
       "Y": "TYR", "V": "VAL"}


_A14 = None
def _atom14_names():
    global _A14
    if _A14 is None:
        from openfold.np import residue_constants as rc
        _A14 = rc.restype_name_to_atom14_names
    return _A14


def write_pdb(path, coords, mask, seq=None):
    """Write a structure. For 4-atom coords -> backbone only; for 14-atom (atom14)
    coords -> full all-atom (sidechains) using the per-residue atom14 names. Residue
    identities come from `seq` (1-letter)."""
    n_atom = coords.shape[1]
    a14 = _atom14_names() if n_atom == 14 else None
    with open(path, "w") as f:
        a = 1
        for ri in range(coords.shape[0]):
            resn = AA3.get(seq[ri], "UNK") if seq and ri < len(seq) else "GLY"
            names = a14[resn] if (a14 and resn in a14) else (BB_NAMES + [""] * (n_atom - 4))
            for ai in range(n_atom):
                an = names[ai] if ai < len(names) else ""
                if not an or not mask[ri, ai]:
                    continue
                x, y, z = coords[ri, ai].tolist()
                f.write(f"ATOM  {a:5d}  {an:<3s} {resn} A{ri+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {an[0]}\n")
                a += 1
        f.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./checkpoints/sabdab/abdiff_final_v1.pt")
    ap.add_argument("--data", default="./data/train_sabdab_cdr")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write-pdb", default="")
    ap.add_argument("--dump", default="")   # JSON of per-region per-example RMSDs (for CI bars)
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck.get("args", {})
    model = AbDiffusion(c_esm=ck.get("c_esm", 512), c=a.get("c", 384), c_z=a.get("c_z", 128), n_head=a.get("n_head", 12), n_block=a.get("n_block", 8)).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"[eval] ckpt epoch={ck.get('epoch')} val={ck.get('val', float('nan')):.4f} device={dev}")

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    val = files[:max(1, len(files) // 10)][:args.n_eval]
    if args.write_pdb:
        os.makedirs(args.write_pdb, exist_ok=True)

    agg = {}   # fmt -> {region -> [vals]}
    for fp in val:
        rec = torch.load(fp, map_location="cpu")
        if "cdr" not in rec:
            print(f"  [skip] {rec['id']} has no CDR labels (re-prep needed)"); continue
        b = to_dev(collate([rec]), dev)
        gt = b["coords"][0]; m = b["atom_mask"][0]
        cdr = rec["cdr"].to(dev).long(); ht = rec["htype"].to(dev).long()
        gt = gt - gt[:, 1][m[:, 1]].mean(0)[None, None, :]
        best = None
        for _ in range(args.n_samples):
            x = sample_backbone(model, b["emb"], b["residue_index"], b["asym_id"],
                                b["token_mask"], b["atom_mask"], n_steps=args.steps)[0]
            r, Pa = region_rmsds(x, gt, m, cdr, ht)
            # rank samples by CDR-H3 (fall back to all if no H3)
            key = r["h3"] if r["h3"] == r["h3"] else r["all"]
            if best is None or key < best[0]:
                best = (key, r, Pa)
        _, r, Pa = best
        f = rec["format"]
        d = agg.setdefault(f, {})
        for k, v in r.items():
            if v == v:  # not NaN
                d.setdefault(k, []).append(v)
        print(f"  {rec['id']:13s} {f:4s} N={gt.shape[0]:4d}  "
              f"all={r['all']:.2f} fr={r['fr']:.2f} H3={r['h3'] if r['h3']==r['h3'] else float('nan'):.2f} "
              f"H3len={int(((cdr==3)&(ht==1)).sum())}")
        if args.write_pdb:
            sq = rec.get("seq")
            write_pdb(os.path.join(args.write_pdb, f"{rec['id']}_pred.pdb"), Pa.cpu(), m.cpu(), sq)
            write_pdb(os.path.join(args.write_pdb, f"{rec['id']}_true.pdb"), gt.cpu(), m.cpu(), sq)

    print("\n[eval] mean RMSD (Å), framework-superposed:")
    print(f"  {'fmt':5s} {'n':>4s} {'all':>6s} {'fr':>6s} {'cdr1':>6s} {'cdr2':>6s} {'cdr3':>6s} {'H3':>6s} {'L3':>6s}")
    for f, d in sorted(agg.items()):
        n = len(d.get("all", []))
        def mu(k): return (sum(d[k]) / len(d[k])) if d.get(k) else float("nan")
        print(f"  {f:5s} {n:4d} {mu('all'):6.2f} {mu('fr'):6.2f} {mu('cdr1'):6.2f} "
              f"{mu('cdr2'):6.2f} {mu('cdr3'):6.2f} {mu('h3'):6.2f} {mu('l3'):6.2f}")
    # overall H3
    allh3 = [v for d in agg.values() for v in d.get("h3", [])]
    if allh3:
        print(f"[eval] *** overall CDR-H3 Cα-RMSD = {sum(allh3)/len(allh3):.2f} Å "
              f"(n={len(allh3)}, steps={args.steps}, best-of-{args.n_samples}) ***")
    if args.dump:
        import json
        json.dump({f: {k: list(map(float, v)) for k, v in d.items()} for f, d in agg.items()},
                  open(args.dump, "w"))
        print(f"[eval] dumped per-example RMSDs -> {args.dump}")


if __name__ == "__main__":
    main()
