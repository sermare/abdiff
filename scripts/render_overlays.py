#!/usr/bin/env python
"""
Render predicted-vs-native antibody overlays with PyMOL (ray-traced cartoon),
CDR-H3 highlighted, framework-superposed. Replaces the matplotlib Cα traces.

Usage:
  python scripts/render_overlays.py --pdb-dir <dir with id_pred.pdb/id_true.pdb> \
      --truth <bench_truth.pt> --ids 11hb_DC 10or_GH ...
Outputs assets/overlay_<id>.png
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")


def resid_lists(t):
    cdr = np.array(t["cdr"]); ht = np.array(t["htype"])
    h3 = [i + 1 for i in range(len(cdr)) if cdr[i] == 3 and ht[i] == 1]
    fr = [i + 1 for i in range(len(cdr)) if cdr[i] == 0]
    return fr, h3


def sel(resids):
    return "resi " + "+".join(str(r) for r in resids) if resids else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--out-prefix", default="overlay_")
    args = ap.parse_args()

    import torch
    truth = torch.load(args.truth, map_location="cpu")

    import pymol
    pymol.pymol_argv = ["pymol", "-qc"]   # quiet, no GUI
    pymol.finish_launching()
    from pymol import cmd

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)

    for hid in args.ids:
        pp = os.path.join(args.pdb_dir, f"{hid}_pred.pdb")
        tp = os.path.join(args.pdb_dir, f"{hid}_true.pdb")
        if hid not in truth or not (os.path.exists(pp) and os.path.exists(tp)):
            print(f"[skip] {hid}"); continue
        fr, h3 = resid_lists(truth[hid])
        cmd.reinitialize()
        cmd.bg_color("white"); cmd.set("ray_opaque_background", 0); cmd.set("antialias", 2)
        cmd.set("ray_shadows", 0)
        cmd.load(tp, "native")
        cmd.load(pp, "abdiff")
        cmd.hide("everything")
        # framework superposition (sequence-independent, CA)
        try:
            cmd.super(f"abdiff and name CA and ({sel(fr)})", f"native and name CA and ({sel(fr)})")
        except Exception:
            cmd.super("abdiff and name CA", "native and name CA")
        cmd.dss()  # assign secondary structure for cartoon
        cmd.show("cartoon", "native or abdiff")
        cmd.set("cartoon_transparency", 0.35, "native")
        cmd.color("grey70", "native")
        cmd.color("marine", "abdiff")
        # CDR-H3 highlight
        cmd.color("firebrick", f"abdiff and ({sel(h3)})")
        cmd.color("orange", f"native and ({sel(h3)})")
        cmd.orient("native")
        cmd.zoom("native", 3)
        out = os.path.join(ASSETS, f"{args.out_prefix}{hid}.png")
        cmd.ray(args.width, args.height)
        cmd.png(out, dpi=150)
        print(f"wrote {out}  (H3 residues: {len(h3)})")


if __name__ == "__main__":
    main()
