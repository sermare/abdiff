#!/usr/bin/env python
"""
Stage 1 (tttppi env, has ANARCI): build the held-out 'truth' bundle for benchmarking
external folders. For the exact ids our model held out, derive ANARCI-trimmed Fv
seqs + backbone coords + per-residue CDR/heavy labels and dump to one file.

Output: data/bench_truth.pt = { id: {"seqs":[...], "coords":[N,4,3], "mask":[N,4],
                                     "cdr":[N], "htype":[N], "format":str} }
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from prep_structures import parse_chain_backbone, get_embedder, embed
from prep_structures_anarci import anarci_domains, cdr_labels_for_domain


def build_truth(arr, entry, runner, dev):
    parsed = []
    for ch in entry["chains"]:
        p = parse_chain_backbone(arr, ch["name"])
        if p is None:
            return None
        parsed.append((ch["name"], *p))
    domains = anarci_domains([(nm, sq) for nm, sq, _, _ in parsed])
    seqs, coords, masks, cdr, htype = [], [], [], [], []
    embs, asym, residx, ent = [], [], [], []
    asym_idx = 0
    for ci, (nm, seq, xyz, mask) in enumerate(parsed):
        spans = domains[ci] or [("?", 0, len(seq) - 1, None)]
        ridx = 0
        for (ct, s, e, numbering) in spans:
            s = max(0, s); e = min(len(seq) - 1, e); e = min(e, s + 509)
            if e - s + 1 < 30:
                continue
            cmap = cdr_labels_for_domain(numbering, s, e - s + 1) if numbering else {}
            L = e - s + 1
            sub = seq[s:e + 1]
            seqs.append(sub); coords.append(xyz[s:e + 1]); masks.append(mask[s:e + 1])
            embs.append(embed(runner, sub, dev).numpy())
            cdr += [cmap.get(s + k, 0) for k in range(L)]
            htype += [1 if ct == "H" else 0] * L
            asym += [asym_idx] * L; ent += [asym_idx] * L
            residx += list(range(ridx, ridx + L)); ridx += L
        asym_idx += 1
    if not seqs:
        return None
    return {"seqs": seqs, "coords": np.concatenate(coords, 0),
            "mask": np.concatenate(masks, 0), "cdr": np.array(cdr),
            "htype": np.array(htype),
            "emb": np.concatenate(embs, 0).astype("float16"),
            "asym_id": np.array(asym), "residue_index": np.array(residx),
            "entity_id": np.array(ent), "format": entry["format"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/train_sabdab")
    ap.add_argument("--corpus", default="./data/ab_corpus_sabdab.jsonl")
    ap.add_argument("--cif-dir", default="./data/cif")
    ap.add_argument("--out", default="./data/bench_truth.pt")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc
    runner = get_embedder(args.device)

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    n_val = max(1, len(files) // 10)
    held_ids = [os.path.basename(f)[:-3] for f in files[:n_val]][:args.n_eval]
    corpus = {}
    for l in open(args.corpus):
        if l.strip():
            r = json.loads(l); corpus[r["id"]] = r

    out = {}
    for hid in held_ids:
        e = corpus.get(hid)
        if e is None:
            continue
        cif = os.path.join(args.cif_dir, f"{e['pdb_id'].lower()}.cif")
        if not os.path.exists(cif):
            continue
        try:
            arr = get_structure(PDBxFile.read(cif), model=1)
            arr = arr[struc.filter_amino_acids(arr)]
            t = build_truth(arr, e, runner, args.device)
        except Exception as ex:
            print(f"[skip] {hid}: {ex}", flush=True); continue
        if t:
            out[hid] = t
    torch.save(out, args.out)
    print(f"[truth] wrote {len(out)} held-out entries -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
