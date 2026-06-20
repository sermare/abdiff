#!/usr/bin/env python
"""Three/four-model comparison figures with 95% CI bars, computed from per-example
RMSD dumps. All numbers are framework-superposed Cα-RMSD (Å) on the held-out set."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
REG = ["fr", "cdr1", "cdr2", "cdr3", "l3", "h3"]
LAB = ["Framework", "CDR1", "CDR2", "CDR3", "CDR-L3", "CDR-H3"]

# (label, per-example JSON, color) — JSON schema {fmt: {region: [vals]}}
SOURCES = [
    ("Boltz-2 (MSA)",          "boltz_perregion.json", "#444444"),
    ("Boltz-2 (no MSA)",       "boltz_ss_perregion.json", "#8a8a8a"),
    ("AbDiff all-atom (ours)", "aa_perexample.json",   "#1f6fb2"),
    ("AbDiff backbone (ours)", "indist_perexample.json", "#9bbbd4"),
]


def load_stats(fname):
    p = os.path.join(ASSETS, fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for r in REG + ["h3"]:
        vals = [v for fmt in d.values() for v in fmt.get(r, [])]
        if vals:
            a = np.array(vals)
            out[r] = (a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0, len(a))
    return out


def present():
    return [(lab, s, col) for (lab, f, col) in SOURCES if (s := load_stats(f)) is not None]


def fig_per_cdr():
    models = present()
    x = np.arange(len(REG)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (lab, st, col) in enumerate(models):
        off = (i - (len(models) - 1) / 2) * w
        means = [st[r][0] if r in st else np.nan for r in REG]
        cis = [st[r][1] if r in st else 0 for r in REG]
        ax.bar(x + off, means, w, yerr=cis, capsize=2, label=lab, color=col,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.9, ecolor="#222"))
    ax.axhline(1.0, ls="--", c="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(LAB)
    ax.set_ylabel("Backbone Cα-RMSD (Å)  (mean ± 95% CI)")
    ax.set_title("Per-CDR accuracy (framework-superposed, held-out)\nAbDiff is single-sequence")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "per_cdr_three_models.png"), dpi=150)
    print("wrote per_cdr_three_models.png")


def fig_h3():
    models = present()
    names = [m[0] for m in models]
    means = [m[1]["h3"][0] for m in models]; cis = [m[1]["h3"][1] for m in models]
    cols = [m[2] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = ax.bar(range(len(names)), means, yerr=cis, capsize=4, color=cols,
               edgecolor="black", linewidth=0.7, width=0.62, error_kw=dict(lw=1.1, ecolor="#222"))
    for bar, mn, ci in zip(b, means, cis):
        ax.text(bar.get_x()+bar.get_width()/2, mn+ci+0.05, f"{mn:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.set_ylabel("CDR-H3 Cα-RMSD (Å)  (mean ± 95% CI)")
    ax.set_ylim(0, max(m + c for m, c in zip(means, cis)) * 1.2)
    ax.set_title("CDR-H3 accuracy — the metric that matters (held-out)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "cdrh3_three_models.png"), dpi=150)
    print("wrote cdrh3_three_models.png")


def fig_perf_allatom():
    """All-atom replica of performance_by_region.png: backbone vs all-atom, per region, ±95% CI."""
    bk = load_stats("indist_perexample.json"); aa = load_stats("aa_perexample.json")
    if not (bk and aa):
        print("skip all-atom perf fig"); return
    groups = [("AbDiff backbone", bk, "#9bbbd4"), ("AbDiff all-atom", aa, "#1f6fb2")]
    x = np.arange(len(REG)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (lab, st, col) in enumerate(groups):
        off = (i - 0.5) * w
        means = [st[r][0] if r in st else np.nan for r in REG]
        cis = [st[r][1] if r in st else 0 for r in REG]
        bars = ax.bar(x + off, means, w, yerr=cis, capsize=3, label=lab, color=col,
                      edgecolor="black", linewidth=0.6, error_kw=dict(lw=1, ecolor="#333"))
        for bar, mn in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, mn+0.05, f"{mn:.1f}", ha="center", fontsize=7)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(LAB)
    ax.set_ylabel("Backbone Cα-RMSD (Å)  (mean ± 95% CI)")
    ax.set_title("AbDiff all-atom vs backbone — RMSD by region\nall-atom improves the CDRs, esp. CDR-H3")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "performance_by_region_allatom.png"), dpi=150)
    print("wrote performance_by_region_allatom.png")


def fig_params_vs_rmsd():
    """Model size (total params) vs CDR-H3 RMSD, with 95% CI. Smaller-left-lower is better."""
    # (label, json, params, color, marker)
    pts = [
        ("AbDiff backbone",   "indist_perexample.json", 14.5e6, "#9bbbd4", "o"),
        ("AbDiff all-atom",   "aa_perexample.json",      14.5e6, "#1f6fb2", "o"),
        ("Boltz-2 (no MSA)",  "boltz_ss_perregion.json", 521e6,  "#8a8a8a", "s"),
        ("Boltz-2 (MSA)",     "boltz_perregion.json",    521e6,  "#444444", "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for lab, fj, par, col, mk in pts:
        st = load_stats(fj)
        if not st or "h3" not in st:
            continue
        m, ci, _ = st["h3"]
        ax.errorbar(par/1e6, m, yerr=ci, fmt=mk, ms=11, color=col, capsize=4,
                    ecolor=col, elinewidth=1.4, label=lab, zorder=3)
        ax.annotate(f"{lab}\n{m:.2f} Å", (par/1e6, m), textcoords="offset points",
                    xytext=(12, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (millions, log scale)")
    ax.set_ylabel("CDR-H3 Cα-RMSD (Å)  (mean ± 95% CI)")
    ax.set_title("Model size vs CDR-H3 accuracy\nAbDiff matches single-seq Boltz-2 at ~36× fewer parameters")
    ax.set_xlim(8, 1500); ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "params_vs_rmsd.png"), dpi=150)
    print("wrote params_vs_rmsd.png")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    fig_per_cdr(); fig_h3(); fig_perf_allatom(); fig_params_vs_rmsd()
