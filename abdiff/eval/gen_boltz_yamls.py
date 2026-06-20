#!/usr/bin/env python
"""Generate Boltz-2 input YAMLs from a truth bundle (one per held-out antibody).
Each Fv chain becomes a protein entry; MSAs come from --use_msa_server at predict time."""
import argparse, os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="./data/bench_truth.pt")
    ap.add_argument("--out", default="./data/boltz_in")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single-seq", action="store_true",
                    help="write 'msa: empty' (no MSA) — fair single-sequence comparison")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    truth = torch.load(args.truth, map_location="cpu", weights_only=False)
    ids = list(truth.keys())
    if args.limit:
        ids = ids[:args.limit]
    chain_ids = "ABCDEFGH"
    n = 0
    for hid in ids:
        seqs = truth[hid]["seqs"]
        lines = ["version: 1", "sequences:"]
        for i, s in enumerate(seqs):
            lines += [f"  - protein:", f"      id: {chain_ids[i]}", f"      sequence: {s}"]
            if args.single_seq:
                lines += ["      msa: empty"]
        open(os.path.join(args.out, f"{hid}.yaml"), "w").write("\n".join(lines) + "\n")
        n += 1
    print(f"[boltz-yaml] wrote {n} YAMLs -> {args.out}")

if __name__ == "__main__":
    main()
