#!/usr/bin/env python
"""Generate OpenFold3 query JSONs from a truth bundle (one per held-out antibody)."""
import argparse, json, os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="./data/bench_truth.pt")
    ap.add_argument("--out", default="./data/of3_in")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    truth = torch.load(args.truth, map_location="cpu", weights_only=False)
    ids = list(truth.keys())
    if args.limit:
        ids = ids[:args.limit]
    cids = "ABCDEFGH"
    queries = {}
    for hid in ids:
        seqs = truth[hid]["seqs"]
        queries[hid] = {"chains": [{"molecule_type": "protein", "chain_ids": [cids[i]],
                                    "sequence": s} for i, s in enumerate(seqs)]}
    out_json = os.path.join(args.out, "queries.json")
    json.dump({"queries": queries}, open(out_json, "w"), indent=2)
    print(f"[of3] wrote {len(queries)} queries -> {out_json}")

if __name__ == "__main__":
    main()
