#!/usr/bin/env python
"""
Build aligned (pLM-embedding, backbone-coordinate) training tensors for AbDiff.

For each entry in data/ab_corpus.jsonl:
  - fetch the mmCIF from RCSB (cached locally), parse model 1
  - for each antibody chain in the entry (PDB chain id = corpus chain name):
      * extract per-residue backbone atoms N, CA, C, O  (in observed order)
      * derive the 1-letter sequence from the observed residues
      * embed that sequence with ESM2 (per-residue, last hidden state)
  - concatenate chains token-wise → per-entry tensors and save:

  data/train/<id>.pt = {
     "emb":   float16 [N_tok, c_esm],     # pLM per-residue embedding
     "coords":float32 [N_tok, 4, 3],      # backbone N,CA,C,O
     "atom_mask": bool [N_tok, 4],        # present atoms
     "asym_id":  int16 [N_tok],           # chain instance
     "residue_index": int32 [N_tok],      # position within chain (observed order)
     "entity_id": int16 [N_tok],          # = asym_id here
     "format": "fab"|"vhh"|"scfv",
     "id": str,
  }

Coordinates are the diffusion targets; everything else conditions the denoiser.
GPU recommended (pLM). Run via sbatch. Compute nodes have outbound network for RCSB.
"""
import argparse, json, os, sys
import numpy as np
import torch

BB = ["N", "CA", "C", "O"]

def log(*a): print(*a, flush=True)

def load_corpus(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def get_embedder(device):
    """AntiBERTy — antibody-specific pLM (26M, 512-d per-residue embeddings)."""
    from antiberty import AntiBERTyRunner
    runner = AntiBERTyRunner()
    try:
        runner.device = torch.device(device)
        runner.model = runner.model.to(device)
    except Exception:
        pass
    log(f"[plm] AntiBERTy ready (device={device}, dim=512)")
    return runner

@torch.no_grad()
def embed(runner, seq, device):
    e = runner.embed([seq])[0]                     # [L+2, 512] (CLS + residues + SEP)
    return e[1:1 + len(seq)].to(torch.float16).cpu()

def parse_chain_backbone(atom_array, chain_id):
    """Return (seq:str, coords:[L,4,3] float32, mask:[L,4] bool) for one chain."""
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence
    sub = atom_array[atom_array.chain_id == chain_id]
    if sub.array_length() == 0:
        return None
    seq_chars, coords, masks = [], [], []
    for res in struc.residue_iter(sub):
        resname = res.res_name[0]
        try:
            one = ProteinSequence.convert_letter_3to1(resname)
        except Exception:
            one = "X"
        if one == "X":           # skip non-standard / hetero
            continue
        xyz = np.zeros((4, 3), dtype=np.float32)
        m = np.zeros(4, dtype=bool)
        for i, an in enumerate(BB):
            sel = res[res.atom_name == an]
            if sel.array_length() >= 1:
                xyz[i] = sel.coord[0]
                m[i] = True
        if not m[1]:             # require CA
            continue
        seq_chars.append(one); coords.append(xyz); masks.append(m)
    if not seq_chars:
        return None
    return "".join(seq_chars), np.stack(coords), np.stack(masks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./data/ab_corpus.jsonl")
    ap.add_argument("--out-dir", default="./data/train")
    ap.add_argument("--cif-dir", default="./data/cif")
    ap.add_argument("--model", default="antiberty")  # antibody pLM
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-chain-len", type=int, default=510)  # AntiBERTy 512-pos limit
    ap.add_argument("--formats", nargs="+", default=["fab", "vhh", "scfv"])
    ap.add_argument("--limit", type=int, default=0)  # 0 = all; for smoke tests
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cif_dir, exist_ok=True)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    from biotite.database import rcsb
    from biotite.structure.io.pdbx import PDBxFile, get_structure
    import biotite.structure as struc

    runner = get_embedder(dev)
    corpus = load_corpus(args.corpus)
    corpus = [e for e in corpus if e["format"] in args.formats]
    if args.limit:
        corpus = corpus[:args.limit]
    log(f"[prep] {len(corpus)} entries, device={dev}")

    ok = fail = 0
    for ei, entry in enumerate(corpus):
        out_p = os.path.join(args.out_dir, f"{entry['id']}.pt")
        if os.path.exists(out_p):
            ok += 1; continue
        pdb = entry["pdb_id"].lower()
        try:
            cif_p = os.path.join(args.cif_dir, f"{pdb}.cif")
            if not os.path.exists(cif_p):
                rcsb.fetch(pdb, "cif", args.cif_dir)
            arr = get_structure(PDBxFile.read(cif_p), model=1)
            arr = arr[struc.filter_amino_acids(arr)]
        except Exception as e:
            log(f"[skip] {entry['id']} fetch/parse: {e}"); fail += 1; continue

        embs, coords, masks, asym, residx, ent = [], [], [], [], [], []
        bad = False
        for ch in entry["chains"]:
            name = ch["name"]
            parsed = parse_chain_backbone(arr, name)
            if parsed is None:
                bad = True; break
            seq, xyz, m = parsed
            if len(seq) > args.max_chain_len:
                seq = seq[:args.max_chain_len]; xyz = xyz[:args.max_chain_len]; m = m[:args.max_chain_len]
            e = embed(runner, seq, dev).numpy()
            embs.append(e); coords.append(xyz); masks.append(m)
            asym += [ch["asym_id"]] * len(seq)
            ent  += [ch["asym_id"]] * len(seq)
            residx += list(range(len(seq)))
        if bad or not embs:
            fail += 1; continue

        rec = {
            "emb": torch.from_numpy(np.concatenate(embs, 0)),
            "coords": torch.from_numpy(np.concatenate(coords, 0)).float(),
            "atom_mask": torch.from_numpy(np.concatenate(masks, 0)),
            "asym_id": torch.tensor(asym, dtype=torch.int16),
            "residue_index": torch.tensor(residx, dtype=torch.int32),
            "entity_id": torch.tensor(ent, dtype=torch.int16),
            "format": entry["format"], "id": entry["id"],
        }
        torch.save(rec, out_p)
        ok += 1
        if (ei + 1) % 25 == 0:
            log(f"[prep] {ei+1}/{len(corpus)} ok={ok} fail={fail}")
    log(f"[prep] DONE ok={ok} fail={fail} -> {args.out_dir}")

if __name__ == "__main__":
    main()
