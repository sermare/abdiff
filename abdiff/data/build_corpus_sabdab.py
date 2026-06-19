#!/usr/bin/env python
"""
Build the AbDiff corpus from the full SAbDab summary table.

SAbDab pre-annotates chains, so format classification is direct:
  Hchain & Lchain present, different ids -> fab  (paired Fv/Fab)
  Hchain present, Lchain == NA           -> vhh  (heavy-only / nanobody / HCAb)
  Hchain == Lchain (same id)             -> scfv (both V-domains on one chain)

Emits the same jsonl schema as build_corpus.py (prep_structures derives the
sequence from the structure, so we only need chain ids + format here):
  {id, pdb_id, format, chains:[{name, asym_id, kind}], antigen:[chain_ids], source}

Quality cut: resolution <= --max-res (when numeric) and standard methods.
One entry per unique (pdb, Hchain, Lchain) antibody.
"""
import argparse, csv, json, os
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="./data/sabdab_summary_all.tsv")
    ap.add_argument("--out", default="./data/ab_corpus_sabdab.jsonl")
    ap.add_argument("--max-res", type=float, default=3.5)
    args = ap.parse_args()

    seen = set()
    fmt = Counter()
    rows_out = []
    with open(args.tsv) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            pdb = (r.get("pdb") or "").strip().lower()
            H = (r.get("Hchain") or "NA").strip()
            L = (r.get("Lchain") or "NA").strip()
            if not pdb or H in ("", "NA"):
                # require a heavy chain (covers fab, vhh, scfv); skip light-only/odd rows
                if L in ("", "NA"):
                    continue
            # resolution filter (keep when parseable)
            res = (r.get("resolution") or "").strip()
            try:
                if res and res.upper() != "NA" and float(res.split(",")[0]) > args.max_res:
                    continue
            except ValueError:
                pass
            method = (r.get("method") or "").upper()
            if method and not any(m in method for m in ("X-RAY", "ELECTRON", "EM", "NMR")):
                continue

            ag = (r.get("antigen_chain") or "NA").strip()
            antigen = [c.strip() for c in ag.replace("|", " ").split() if c.strip() not in ("", "NA")]

            if H not in ("", "NA") and L not in ("", "NA") and H != L:
                format_, chains = "fab", [{"name": H, "asym_id": 0, "kind": "heavy"},
                                          {"name": L, "asym_id": 1, "kind": "light"}]
            elif H not in ("", "NA") and (L in ("", "NA")):
                format_, chains = "vhh", [{"name": H, "asym_id": 0, "kind": "heavy"}]
            elif H != "" and H == L:
                format_, chains = "scfv", [{"name": H, "asym_id": 0, "kind": "scfv"}]
            else:
                continue

            key = (pdb, H, L)
            if key in seen:
                continue
            seen.add(key)
            fmt[format_] += 1
            rows_out.append({"id": f"{pdb}_{H}{L if L not in ('','NA') else ''}",
                             "pdb_id": pdb, "format": format_, "chains": chains,
                             "antigen": antigen, "source": "sabdab_summary"})

    with open(args.out, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    pdbs = {r["pdb_id"] for r in rows_out}
    print(f"[sabdab] entries={len(rows_out)} unique_pdb={len(pdbs)} formats={dict(fmt)}")
    print(f"[sabdab] wrote {args.out}")
    # also dump the unique pdb id list for prefetch
    idp = os.path.join(os.path.dirname(args.out), "sabdab_pdb_ids.txt")
    with open(idp, "w") as f:
        f.write("\n".join(sorted(pdbs)) + "\n")
    print(f"[sabdab] wrote {idp} ({len(pdbs)} ids)")

if __name__ == "__main__":
    main()
