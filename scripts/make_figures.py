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


def _stats(perex):
    """Pool all formats per region -> (mean, 95% CI half-width). perex={fmt:{region:[vals]}}."""
    out = {}
    for r in REG:
        vals = [v for fmt in perex.values() for v in fmt.get(r, [])]
        if not vals:
            out[r] = (float("nan"), 0.0); continue
        a = np.array(vals); m = a.mean()
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        out[r] = (m, ci)
    return out


def fig_performance(indist, ood=None):
    di = _stats(indist)
    groups = [("in-distribution held-out", di, "#5b8fb0")]
    if ood:
        groups.append(("out-of-distribution (never-trained PDBs)", _stats(ood), "#e08a3c"))
    x = np.arange(len(REG)); w = 0.38 if ood else 0.6
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (lab, st, col) in enumerate(groups):
        off = (i - (len(groups) - 1) / 2) * w
        means = [st[r][0] for r in REG]; cis = [st[r][1] for r in REG]
        bars = ax.bar(x + off, means, w, yerr=cis, capsize=3, label=lab, color=col,
                      edgecolor="black", linewidth=0.6, error_kw=dict(lw=1, ecolor="#333"))
        for b, mn in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mn + max(means) * 0.02 + 0.06,
                    f"{mn:.1f}", ha="center", fontsize=7.5)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(REGLAB)
    ax.set_ylabel("Backbone Cα-RMSD (Å)  (mean ± 95% CI)")
    ax.set_title("AbDiff — framework-superposed RMSD by region\n"
                 "framework ~1 Å; CDR-H3 is the hard loop; OOD≈in-dist ⇒ generalizes")
    ax.legend(fontsize=8, frameon=False)
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
    ap.add_argument("--indist", default=os.path.join(ASSETS, "indist_perexample.json"))
    ap.add_argument("--ood", default=os.path.join(ASSETS, "ood_perexample.json"))
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    indist = json.load(open(args.indist))
    ood = json.load(open(args.ood)) if os.path.exists(args.ood) else None
    fig_performance(indist, ood)
    fig_architecture()


if __name__ == "__main__":
    main()
