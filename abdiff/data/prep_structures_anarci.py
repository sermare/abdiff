#!/usr/bin/env python
"""
ANARCI-based structure prep for the full SAbDab corpus.

For each entry (PDB + Hchain/Lchain ids from the SAbDab summary):
  - parse model 1, extract per-residue backbone (N,CA,C,O) + structure-derived seq
  - run ANARCI (IMGT) on each chain -> variable-domain (Fv) boundaries + chain type
      * 1 domain  -> H => heavy(VHH if sole chain) ; K/L => light
      * 2 domains on ONE chain -> scFv (VH + VL); keep both, drop the linker
  - TRIM seq+coords to the Fv domain(s); concatenate domains/chains token-wise
  - AntiBERTy-embed the trimmed Fv sequence(s)
  - save data/train_sabdab/<id>.pt  (same schema as prep_structures.py, plus per-token
    'domain_id' so VH/VL within an scFv are distinguishable)

Format is decided from ANARCI (overrides the summary): scFv if any chain has 2 domains;
fab if heavy+light; vhh if heavy-only.

Requires hmmscan on PATH (env bin) — the sbatch prepends it.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from prep_structures import parse_chain_backbone, parse_chain_atom14, get_embedder, embed, BB, log


# hmmscan lives in the env bin but srun does not reliably propagate PATH to the
# step. ANARCI's internal check_for_j() calls run_hmmer WITHOUT honoring hmmerpath,
# so passing hmmerpath is not enough — put the env bin on PATH for ALL subprocesses.
HMMER_BIN = os.environ.get("ABDIFF_HMMER_BIN",
                           os.path.join(os.environ.get("CONDA_PREFIX",""),"bin"))
os.environ["PATH"] = HMMER_BIN + os.pathsep + os.environ.get("PATH", "")

def anarci_domains(named_seqs):
    """named_seqs: list[(id,seq)]. Returns list (parallel) of [(chain_type,start,end),...].
    Returns [] per seq on any ANARCI failure (caller falls back to full chain)."""
    from anarci import anarci
    try:
        num, det, _ = anarci(named_seqs, scheme="imgt", output=False, hmmerpath=HMMER_BIN)
    except Exception:
        return [[] for _ in named_seqs]
    out = []
    for i, doms in enumerate(num):
        if not doms:
            out.append([]); continue
        spans = []
        for j, d in enumerate(doms):
            numbering, start, end = d   # numbering = [((pos,ins),aa), ...]
            ct = det[i][j]["chain_type"]
            spans.append((ct, start, end, numbering))
        out.append(spans)
    return out


def imgt_cdr(pos):
    """IMGT CDR region for a numbering position: 1/2/3 for CDR1/2/3, else 0 (framework)."""
    if 27 <= pos <= 38:  return 1
    if 56 <= pos <= 65:  return 2
    if 105 <= pos <= 117: return 3
    return 0


def cdr_labels_for_domain(numbering, start, n_res):
    """Map a domain's ANARCI numbering -> per-residue CDR label for sequence indices
    start..start+n_res-1 (skipping gap '-' entries). Returns dict {seq_idx: cdr}."""
    out, ri = {}, start
    for (pos, _ins), aa in numbering:
        if aa == "-":
            continue
        out[ri] = imgt_cdr(pos); ri += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./data/ab_corpus_sabdab.jsonl")
    ap.add_argument("--out-dir", default="./data/train_sabdab")
    ap.add_argument("--cif-dir", default="./data/cif")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)    # this shard index
    ap.add_argument("--nshards", type=int, default=1)  # total shards (array job)
    ap.add_argument("--no-trim", action="store_true", help="keep full chains (skip Fv trim)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    from biotite.database import rcsb
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc

    runner = get_embedder(dev)
    corpus = [json.loads(l) for l in open(args.corpus) if l.strip()]
    if args.limit:
        corpus = corpus[:args.limit]
    if args.nshards > 1:
        corpus = corpus[args.shard::args.nshards]   # interleaved shard
    log(f"[anarci-prep] {len(corpus)} entries (shard {args.shard}/{args.nshards}) "
        f"device={dev} trim={not args.no_trim}")

    ok = fail = sk = 0
    fmt_count = {"fab": 0, "vhh": 0, "scfv": 0}
    for ei, entry in enumerate(corpus):
        out_p = os.path.join(args.out_dir, f"{entry['id']}.pt")
        if os.path.exists(out_p):
            ok += 1; continue
        pdb = entry["pdb_id"].lower()
        cif_p = os.path.join(args.cif_dir, f"{pdb}.cif")
        try:
            if not os.path.exists(cif_p):
                rcsb.fetch(pdb, "cif", args.cif_dir)
            arr = get_structure(PDBxFile.read(cif_p), model=1)
            arr = arr[struc.filter_amino_acids(arr)]
        except Exception as e:
            fail += 1
            if fail <= 20: log(f"[skip] {entry['id']} parse: {e}")
            continue

        # parse each annotated antibody chain
        parsed = []  # (chain_name, seq, xyz, mask)
        bad = False
        for ch in entry["chains"]:
            p = parse_chain_atom14(arr, ch["name"])   # all-atom (atom14); CA at index 1
            if p is None:
                bad = True; break
            parsed.append((ch["name"], *p))
        if bad or not parsed:
            fail += 1; continue

        # ANARCI on all chains at once
        domains = ([[] for _ in parsed] if args.no_trim
                   else anarci_domains([(nm, sq) for nm, sq, _, _ in parsed]))

        embs, coords, masks, asym, residx, ent, domid = [], [], [], [], [], [], []
        cdr, htype = [], []   # per-token: CDR region (0/1/2/3) and heavy-domain flag
        seqstr = []           # per-token 1-letter residue (for all-atom PDB output)
        n_var_domains = 0
        asym_idx = 0
        for ci, (nm, seq, xyz, mask) in enumerate(parsed):
            spans = domains[ci] if not args.no_trim else []
            if not spans:
                # ANARCI found nothing -> keep full chain as one segment (fallback)
                segs = [("?", 0, len(seq) - 1, None)]
            else:
                segs = list(spans)
                n_var_domains += len(segs)
            ridx = 0
            for di, (ct, s, e, numbering) in enumerate(segs):
                s = max(0, s); e = min(len(seq) - 1, e)
                e = min(e, s + 509)      # AntiBERTy/BERT caps at 512 positions
                sub = seq[s:e + 1]
                if len(sub) < 30:        # too short to be a real Fv
                    continue
                em = embed(runner, sub, dev).numpy()
                embs.append(em)
                coords.append(xyz[s:e + 1]); masks.append(mask[s:e + 1])
                L = e - s + 1
                asym += [asym_idx] * L
                ent += [asym_idx] * L
                residx += list(range(ridx, ridx + L)); ridx += L
                domid += [di] * L
                # per-residue CDR labels (IMGT) + heavy flag
                cmap = cdr_labels_for_domain(numbering, s, L) if numbering else {}
                cdr += [cmap.get(s + k, 0) for k in range(L)]
                htype += [1 if ct == "H" else 0] * L
                seqstr.append(sub)
            asym_idx += 1
        if not embs:
            sk += 1; continue

        # decide format from ANARCI
        if n_var_domains >= 2 and len(parsed) == 1:
            fmt = "scfv"
        elif any(len(domains[ci]) >= 2 for ci in range(len(parsed))) and not args.no_trim:
            fmt = "scfv"
        elif len(parsed) >= 2:
            fmt = "fab"
        else:
            fmt = "vhh"
        fmt_count[fmt] = fmt_count.get(fmt, 0) + 1

        rec = {
            "emb": torch.from_numpy(np.concatenate(embs, 0)),
            "coords": torch.from_numpy(np.concatenate(coords, 0)).float(),
            "atom_mask": torch.from_numpy(np.concatenate(masks, 0)),
            "asym_id": torch.tensor(asym, dtype=torch.int16),
            "residue_index": torch.tensor(residx, dtype=torch.int32),
            "entity_id": torch.tensor(ent, dtype=torch.int16),
            "domain_id": torch.tensor(domid, dtype=torch.int16),
            "seq": "".join(seqstr),                            # 1-letter, token order (for all-atom output)
            "cdr": torch.tensor(cdr, dtype=torch.int8),        # 0=FR,1=CDR1,2=CDR2,3=CDR3
            "htype": torch.tensor(htype, dtype=torch.int8),    # 1=heavy domain
            "format": fmt, "id": entry["id"],
        }
        torch.save(rec, out_p)
        ok += 1
        if (ei + 1) % 100 == 0:
            log(f"[anarci-prep] {ei+1}/{len(corpus)} ok={ok} fail={fail} short_skip={sk} fmt={fmt_count}")
    log(f"[anarci-prep] DONE ok={ok} fail={fail} short_skip={sk} fmt={fmt_count} -> {args.out_dir}")


if __name__ == "__main__":
    main()
