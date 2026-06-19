#!/usr/bin/env python
"""Pre-download mmCIF files for the corpus PDB ids (login node has network).
Pure stdlib so python startup is fast. Skips files already present."""
import json, os, sys, urllib.request

CORPUS = "./data/ab_corpus.jsonl"
OUT = "./data/cif"
URL = "https://files.rcsb.org/download/{}.cif"

def main():
    os.makedirs(OUT, exist_ok=True)
    pdbs = []
    for line in open(CORPUS):
        if line.strip():
            pdbs.append(json.loads(line)["pdb_id"].lower())
    pdbs = sorted(set(pdbs))
    ok = skip = fail = 0
    for i, p in enumerate(pdbs):
        dst = os.path.join(OUT, f"{p}.cif")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1; continue
        try:
            req = urllib.request.Request(URL.format(p.upper()),
                                         headers={"User-Agent": "abdiff/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
                f.write(r.read())
            ok += 1
        except Exception as e:
            fail += 1; print(f"[fail] {p}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[prefetch] {i+1}/{len(pdbs)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[prefetch] DONE total={len(pdbs)} ok={ok} skip={skip} fail={fail} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
