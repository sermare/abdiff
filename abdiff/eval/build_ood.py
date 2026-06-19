#!/usr/bin/env python
"""Build an out-of-distribution test set: SAbDab antibodies we did NOT train on.
Training used resolution <= 3.5 A; here we take entries with resolution > 3.5 A
(a disjoint set of PDBs) and exclude any PDB id present in the trained set."""
import csv, json, os
from collections import Counter

TSV = "./data/sabdab_summary_all.tsv"
TRAINED = "./data/sabdab_pdb_ids.txt"
OUT = "./data/ab_corpus_ood.jsonl"
IDS = "./data/ood_pdb_ids.txt"
MIN_RES, MAX_RES, SAMPLE = 3.5, 6.0, 60

trained = {l.strip() for l in open(TRAINED) if l.strip()}
seen, fmt, rows = set(), Counter(), []
for r in csv.DictReader(open(TSV), delimiter="\t"):
    pdb = (r.get("pdb") or "").strip().lower()
    H = (r.get("Hchain") or "NA").strip(); L = (r.get("Lchain") or "NA").strip()
    if not pdb or pdb in trained or H in ("", "NA"):
        continue
    res = (r.get("resolution") or "").strip()
    try:
        rv = float(res.split(",")[0])
    except ValueError:
        continue
    if not (MIN_RES < rv <= MAX_RES):
        continue
    if H not in ("", "NA") and L not in ("", "NA") and H != L:
        f, chains = "fab", [{"name": H, "asym_id": 0, "kind": "heavy"},
                            {"name": L, "asym_id": 1, "kind": "light"}]
    elif L in ("", "NA"):
        f, chains = "vhh", [{"name": H, "asym_id": 0, "kind": "heavy"}]
    else:
        continue
    key = (pdb, H, L)
    if key in seen:
        continue
    seen.add(key); fmt[f] += 1
    rows.append({"id": f"{pdb}_{H}{L if L not in ('','NA') else ''}", "pdb_id": pdb,
                 "format": f, "chains": chains, "resolution": rv, "source": "sabdab_ood"})

# diversify: sort by pdb, take a spread of fab + vhh
rows.sort(key=lambda r: r["pdb_id"])
fabs = [r for r in rows if r["format"] == "fab"]
vhhs = [r for r in rows if r["format"] == "vhh"]
pick = fabs[::max(1, len(fabs)//40)][:40] + vhhs[::max(1, len(vhhs)//20)][:20]
pick = pick[:SAMPLE]
with open(OUT, "w") as fo:
    for r in pick:
        fo.write(json.dumps(r) + "\n")
with open(IDS, "w") as fi:
    fi.write("\n".join(sorted({r["pdb_id"] for r in pick})) + "\n")
print(f"[ood] candidates: {dict(fmt)} (total {len(rows)}); picked {len(pick)} "
      f"({sum(r['format']=='fab' for r in pick)} fab / {sum(r['format']=='vhh' for r in pick)} vhh)")
print(f"[ood] resolution range {MIN_RES}-{MAX_RES} A, disjoint from {len(trained)} trained PDBs")
print(f"[ood] wrote {OUT} and {IDS}")
