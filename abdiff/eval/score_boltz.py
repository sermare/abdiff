#!/usr/bin/env python
"""Score Boltz-2 predictions against the held-out truth bundle with the SAME
framework-superpose -> per-CDR Cα-RMSD protocol used for AbDiff/ESMFold.
Boltz outputs <out>/boltz_results_<indir>/predictions/<id>/<id>_model_0.cif."""
import argparse, glob, os, sys
import numpy as np
import torch
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from bench_esmfold import score   # identical kabsch + per-region scoring (CA at index 1)

BB = ["N", "CA", "C", "O"]


def cif_backbone(path):
    """Return [Nres,4,3] backbone in residue order (chains concatenated in file order)."""
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc
    arr = get_structure(PDBxFile.read(path), model=1)
    arr = arr[struc.filter_amino_acids(arr)]
    coords = []
    for res in struc.residue_iter(arr):
        xyz = np.zeros((4, 3), np.float32)
        for i, an in enumerate(BB):
            sel = res[res.atom_name == an]
            if sel.array_length() >= 1:
                xyz[i] = sel.coord[0]
        coords.append(xyz)
    return np.stack(coords) if coords else np.zeros((0, 4, 3), np.float32)


def find_cif(out_dir, hid):
    hits = glob.glob(os.path.join(out_dir, "**", f"{hid}_model_0.cif"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(out_dir, "**", hid, "*_model_0.cif"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="./data/bench_truth.pt")
    ap.add_argument("--out-dir", default="./results/boltz_out_ss")
    ap.add_argument("--tag", default="Boltz-2 (single-seq)")
    args = ap.parse_args()
    truth = torch.load(args.truth, map_location="cpu", weights_only=False)
    agg = {}
    for hid, t in truth.items():
        cif = find_cif(args.out_dir, hid)
        if not cif:
            continue
        try:
            pred = cif_backbone(cif)
            gt = torch.tensor(t["coords"]); m = torch.tensor(t["mask"])
            if pred.shape[0] != gt.shape[0]:
                print(f"  [skip] {hid} len pred={pred.shape[0]} gt={gt.shape[0]}"); continue
            gt = gt - gt[:, 1][m[:, 1]].mean(0)[None, None, :]
            r = score(torch.tensor(pred), gt, m, torch.tensor(t["cdr"]).long(),
                      torch.tensor(t["htype"]).long())
        except Exception as e:
            print(f"  [skip] {hid}: {e}"); continue
        f = t["format"]; d = agg.setdefault(f, {})
        for k, v in r.items():
            if v == v:
                d.setdefault(k, []).append(v)
        print(f"  {hid:13s} {f:4s} all={r['all']:.2f} fr={r['fr']:.2f} "
              f"H3={r['h3'] if r['h3']==r['h3'] else float('nan'):.2f}")
    print(f"\n[{args.tag}] mean RMSD (Å), framework-superposed:")
    print(f"  {'fmt':5s} {'n':>4s} {'all':>6s} {'fr':>6s} {'cdr1':>6s} {'cdr2':>6s} {'cdr3':>6s} {'H3':>6s} {'L3':>6s}")
    for f, d in sorted(agg.items()):
        def mu(k): return (sum(d[k]) / len(d[k])) if d.get(k) else float("nan")
        print(f"  {f:5s} {len(d.get('all', [])):4d} {mu('all'):6.2f} {mu('fr'):6.2f} {mu('cdr1'):6.2f} "
              f"{mu('cdr2'):6.2f} {mu('cdr3'):6.2f} {mu('h3'):6.2f} {mu('l3'):6.2f}")
    allh3 = [v for d in agg.values() for v in d.get("h3", [])]
    if allh3:
        print(f"[{args.tag}] *** overall CDR-H3 Cα-RMSD = {sum(allh3)/len(allh3):.2f} Å (n={len(allh3)}) ***")
    import json as _json
    _json.dump({f: {k: list(map(float, v)) for k, v in d.items()} for f, d in agg.items()},
               open("./results/boltz_perregion.json", "w"))


if __name__ == "__main__":
    main()
