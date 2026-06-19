#!/usr/bin/env python
"""Generate the figures used in the README: performance-by-region bar chart,
architecture schematic, and predicted-vs-true Cα overlays (CDR-H3 highlighted)."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
REG = ["fr", "cdr1", "cdr2", "cdr3", "l3", "h3"]
REGLAB = ["Framework", "CDR1", "CDR2", "CDR3", "CDR-L3", "CDR-H3"]


def fig_performance(summary):
    m = summary["models"]["AbDiff (ours, single-seq)"]
    vals = [m[r] for r in REG]
    colors = ["#9bbbd4"] * 5 + ["#d1495b"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(REGLAB, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_ylabel("Backbone Cα-RMSD (Å)")
    ax.set_title("AbDiff — framework-superposed RMSD by region (held-out Fabs)\n"
                 "framework is easy (~1 Å); CDR-H3 is the hard, functional loop")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "performance_by_region.png"), dpi=150)
    print("wrote performance_by_region.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax.axis("off")
    def box(x, y, w, h, txt, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="black", lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9, zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
    box(0.2, 4.3, 2.2, 0.7, "Antibody sequence(s)\nFab / scFv / VHH", "#eef3f8")
    box(0.2, 3.0, 2.2, 0.8, "Frozen pLM (AntiBERTy)\nper-residue 512-d", "#d9e8f5")
    box(0.2, 1.6, 2.2, 0.8, "ANARCI (IMGT)\nFv trim + CDR labels", "#d9e8f5")
    box(3.0, 3.0, 2.2, 0.8, "single rep  s\n(Linear+LN)", "#cfe8d8")
    box(3.0, 1.6, 2.2, 0.8, "pair rep  z\nrelpos + OPM + Pairformer", "#cfe8d8")
    box(6.0, 2.0, 3.0, 2.0,
        "DiffusionModule (R³, EDM)\nno SO(3)/frames\n\nAtomAttn ➜ DiffTransformer\n(pair-bias) ➜ AtomAttn\nx̂ = c_skip·x + c_out·F", "#f6e0c9")
    box(6.6, 4.4, 1.9, 0.6, "noisy coords xₜ\n+ noise level t", "#f9efe1")
    arrow(1.3, 4.3, 1.3, 3.8); arrow(1.3, 3.0, 1.3, 2.4)
    arrow(2.4, 3.4, 3.0, 3.4); arrow(2.4, 2.0, 3.0, 2.0)
    arrow(5.2, 3.4, 6.2, 3.4); arrow(5.2, 2.0, 6.0, 2.4)
    arrow(7.5, 4.4, 7.5, 4.0)
    arrow(9.0, 3.0, 9.7, 3.0); ax.text(9.75, 3.0, "3D backbone\n(N,Cα,C,O)", va="center", fontsize=9)
    ax.set_xlim(0, 11.5); ax.set_ylim(1.2, 5.3)
    ax.set_title("AbDiff: pLM-conditioned R³ coordinate diffusion (AF3/OpenFold3-style, antibody-only)")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "architecture.png"), dpi=150)
    print("wrote architecture.png")


def main():
    # Performance bar chart + architecture schematic only.
    # Structure overlays are rendered with PyMOL (real cartoons) — see scripts/render_overlays.py
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(ASSETS, "bench_summary.json"))
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    fig_performance(json.load(open(args.summary)))
    fig_architecture()


if __name__ == "__main__":
    main()
