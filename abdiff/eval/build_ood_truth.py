#!/usr/bin/env python
"""Build the OOD truth bundle directly from ab_corpus_ood.jsonl (every entry),
reusing the same ANARCI-Fv + AntiBERTy logic as the in-distribution benchmark."""
import json, os, sys
import torch
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
from bench_prep_truth import build_truth
from prep_structures import get_embedder

CORPUS = "./data/ab_corpus_ood.jsonl"
CIF = "./data/cif"
OUT = "./data/ood_truth.pt"

def main():
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc
    runner = get_embedder("cpu")
    out = {}
    for l in open(CORPUS):
        e = json.loads(l)
        cif = os.path.join(CIF, f"{e['pdb_id'].lower()}.cif")
        if not os.path.exists(cif):
            continue
        try:
            arr = get_structure(PDBxFile.read(cif), model=1)
            arr = arr[struc.filter_amino_acids(arr)]
            t = build_truth(arr, e, runner, "cpu")
        except Exception as ex:
            print(f"[skip] {e['id']}: {ex}", flush=True); continue
        if t:
            t["resolution"] = e.get("resolution")
            out[e["id"]] = t
    torch.save(out, OUT)
    print(f"[ood-truth] {len(out)} entries -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
