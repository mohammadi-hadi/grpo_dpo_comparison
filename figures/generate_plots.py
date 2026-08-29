"""Publication figures for the GRPO vs DPO faithfulness study.

Every figure reads from results/ files (no hardcoded data). Figures whose
inputs are missing are skipped, so the script can run at any pipeline stage.

Outputs PDF (vector) sized for a 2-column ACL paper:
column width 3.03 in, full width 6.30 in.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "training_logs"
AGG = ROOT / "results" / "aggregate"
PER_EX = ROOT / "results" / "per_example"
OUT = Path(__file__).resolve().parent

COL_W, FULL_W = 3.03, 6.30

# Colorblind-validated palette (worst adjacent CVD deltaE 21+; sub-3:1 slots
# get direct labels as relief).
C_GRPO = "#2a78d6"   # blue
C_DPO = "#e34948"    # red
QUAD_COLORS = {
    "correct_entailed": "#2a78d6",
    "correct_not_entailed": "#1baf7a",
    "wrong_entailed": "#e34948",       # fluent-but-wrong: the highlight
    "wrong_not_entailed": "#eda100",
}
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal, 4 sizes
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

SIZES = ["1.5b", "3b", "7b", "14b"]
SIZE_X = [1.5, 3, 7, 14]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def fig_training_dynamics():
    files = sorted(LOGS.glob("grpo-*_trainer_state.json"))
    if not files:
        print("skip training_dynamics (no logs)")
        return
    # One distinct CVD-safe color per model size (not a single-hue ramp), a
    # bold exponentially smoothed line (span 5) over the faint raw steps, and
    # both a legend and end labels, per the camera-ready version of Figure 3.
    colors = {"1.5b": "#0072B2", "3b": "#E69F00", "7b": "#009E73", "14b": "#8E4B9E"}
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    order = {"1.5b": 0, "3b": 1, "7b": 2, "14b": 3}
    for f in sorted(files, key=lambda f: order.get(f.name.split("-")[1], 0)):
        size = f.name.split("-")[1]
        hist = json.load(open(f))["log_history"]
        steps = [h["step"] for h in hist if "rewards/check_answer/mean" in h]
        vals = [h["rewards/check_answer/mean"] for h in hist
                if "rewards/check_answer/mean" in h]
        c = colors.get(size, INK)
        smooth = pd.Series(vals).ewm(span=5).mean()
        ax.plot(steps, vals, color=c, lw=0.6, alpha=0.25)
        ax.plot(steps, smooth, color=c, lw=1.6, solid_capstyle="round",
                label=size.replace("b", "B"))
        ax.annotate(size.replace("b", "B"), (steps[-1], smooth.iloc[-1]),
                    textcoords="offset points", xytext=(3, 0),
                    fontsize=6, color=c, va="center")
    style_axis(ax)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Answer-correctness reward", fontsize=6)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.10)  # room for end labels
    ax.legend(loc="lower right", fontsize=5.5, frameon=False,
              handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "training_dynamics.pdf")
    plt.close(fig)
    print("wrote training_dynamics.pdf")


METRIC_PANELS = [
    ("greedy_correct", "Greedy accuracy"),
    ("sc_correct", "Self-consistency"),
    ("consistency_ratio", "Consistency ratio"),
    ("nli_score", "Entailment"),
    ("judge_score_first", "Judge (GPT-5-mini)"),
]


def fig_scaling_curves():
    path = AGG / "aggregate_new.csv"
    if not path.exists():
        print("skip scaling_curves (no aggregate_new.csv)")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 1.75), sharex=True)
    for ax, (metric, title) in zip(axes, METRIC_PANELS):
        for method, color in [("grpo", C_GRPO), ("dpo", C_DPO)]:
            sub = df[df.method == method].set_index("size").reindex(SIZES)
            if metric not in sub.columns or sub[metric].isna().all():
                continue
            y = sub[metric].values
            lo = sub.get(f"{metric}_ci_lo", pd.Series([np.nan] * 4)).values
            hi = sub.get(f"{metric}_ci_hi", pd.Series([np.nan] * 4)).values
            ax.plot(SIZE_X, y, color=color, lw=2, marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=1,
                    label=method.upper())
            if not np.isnan(lo).all():
                ax.fill_between(SIZE_X, lo.astype(float), hi.astype(float),
                                color=color, alpha=0.10, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(SIZE_X, ["1.5", "3", "7", "14"])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8)
        style_axis(ax)
        if ax is axes[0]:
            ax.set_ylabel("Score")
            ax.legend(loc="upper left", fontsize=7, handlelength=1.2)
        else:
            ax.set_yticklabels([])
    axes[2].set_xlabel("Model size (B parameters, log scale)")
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(OUT / "scaling_curves.pdf")
    plt.close(fig)
    print("wrote scaling_curves.pdf")


QUAD_LABELS = {
    "correct_entailed": "correct + entailed",
    "correct_not_entailed": "correct + not entailed",
    "wrong_entailed": "wrong + entailed",
    "wrong_not_entailed": "wrong + not entailed",
}


def quadrant_shares(df, entail_thresh=0.5):
    correct = df["sc_correct"].astype(bool)
    entailed = df["nli_score"] >= entail_thresh
    n = len(df)
    return {
        "correct_entailed": (correct & entailed).sum() / n,
        "correct_not_entailed": (correct & ~entailed).sum() / n,
        "wrong_entailed": (~correct & entailed).sum() / n,
        "wrong_not_entailed": (~correct & ~entailed).sum() / n,
    }


def fig_quadrants():
    frames = {}
    for method in ["dpo", "grpo"]:
        for size in SIZES:
            p = PER_EX / f"{method}_{size}_per_question.csv"
            if p.exists():
                d = pd.read_csv(p)
                if {"sc_correct", "nli_score"}.issubset(d.columns):
                    frames[(method, size)] = quadrant_shares(d)
    if not frames:
        print("skip quadrants (no per-question CSVs)")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    keys = [(m, s) for s in SIZES for m in ["dpo", "grpo"] if (m, s) in frames]
    xpos = np.arange(len(keys), dtype=float)
    # group pairs per size with a gap
    for i in range(len(keys)):
        xpos[i] += (i // 2) * 0.5
    for x, key in zip(xpos, keys):
        bottom = 0.0
        for quad, color in QUAD_COLORS.items():
            v = frames[key][quad]
            ax.bar(x, v, bottom=bottom, width=0.72, color=color,
                   edgecolor="white", linewidth=1)  # surface gap
            if v >= 0.08:  # direct-label relief for low-contrast slots
                ax.text(x, bottom + v / 2, f"{v*100:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if quad in ("correct_entailed",
                                                  "wrong_entailed") else INK)
            bottom += v
    ax.set_xticks(xpos, [f"{m.upper()}\n{s.replace('b','B')}" for m, s in keys],
                  fontsize=6.5)
    ax.set_ylabel("Share of questions")
    ax.set_ylim(0, 1)
    style_axis(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in QUAD_COLORS.values()]
    ax.legend(handles, QUAD_LABELS.values(), fontsize=6, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(OUT / "quadrants.pdf")
    plt.close(fig)
    print("wrote quadrants.pdf")


def fig_judge_agreement():
    """System-level judge agreement: original Gemma-3-27B aggregates (table1)
    vs GPT-5-mini re-evaluation aggregates, one point per fine-tuned model."""
    t1 = AGG / "table1.csv"
    agg = AGG / "aggregate_new.csv"
    if not (t1.exists() and agg.exists()):
        print("skip judge_agreement (missing aggregates)")
        return
    orig = pd.read_csv(t1)
    new = pd.read_csv(agg)
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    for _, r in new.iterrows():
        o = orig[
            (orig.method == r["method"])
            & (orig.size_b == float(r["size"].replace("b", "")))
        ]["llm_judge_gemma3_27b"].iloc[0]
        color = C_GRPO if r["method"] == "grpo" else C_DPO
        ax.scatter(o, r["judge_score_first"], s=42, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2)
        dx, dy = (5, 3) if r["method"] == "grpo" else (-5, -9)
        ax.annotate(r["size"].replace("b", "B"),
                    (o, r["judge_score_first"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="left" if r["method"] == "grpo" else "right",
                    fontsize=6.5, color=INK)
    style_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Gemma 3 27B (original eval)")
    ax.set_ylabel("GPT-5-mini (re-evaluation)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for c, l in [(C_GRPO, "GRPO"), (C_DPO, "DPO")]]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.savefig(OUT / "judge_agreement.pdf")
    plt.close(fig)
    print("wrote judge_agreement.pdf")


if __name__ == "__main__":
    fig_training_dynamics()
    fig_scaling_curves()
    fig_quadrants()
    fig_judge_agreement()
