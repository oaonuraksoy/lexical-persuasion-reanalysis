"""Figures 1-4 and the graphical abstract, drawn from results/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
A, B = RES["corpus_a"], RES["corpus_b"]

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERM, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#7f7f7f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})

SOURCES = ["Control", "Claude 1.3", "Claude Instant 1.2", "Claude 2", "Claude 3 Haiku", "Claude 3 Opus", "Human"]
LABELS = ["Control", "Claude\n1.3", "Claude\nInst. 1.2", "Claude\n2", "Claude 3\nHaiku", "Claude 3\nOpus", "Human"]
COLORS = [GREY] + [SKY] * 4 + [BLUE] + [ORANGE]
FEAT_LABEL = {"collective": "Collective-reference terms", "connectives": "Logical connectives",
              "evidential": "Evidential nouns", "numeric": "Numeric tokens", "hedges": "Hedges and qualifiers",
              "empathy": "Affective terms", "loss": "Loss/threat terms", "certainty": "Certainty terms",
              "n_words": "Argument length", "sent_sd": "Sentence-length SD"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1():
    t = A["table1"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.6), gridspec_kw={"hspace": 0.55})
    x = np.arange(len(SOURCES))
    m = [t[s]["mean"] for s in SOURCES]
    lo = [t[s]["mean"] - t[s]["mean_ci"][0] for s in SOURCES]
    hi = [t[s]["mean_ci"][1] - t[s]["mean"] for s in SOURCES]
    ax1.bar(x, m, color=COLORS, width=0.65)
    ax1.errorbar(x, m, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax1.set_xticks(x, LABELS, fontsize=7.5)
    ax1.set_ylabel("Mean belief shift (Likert points)")
    ax1.set_title("(a) Mean shift, 95% argument-bootstrap CI")
    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.plot(x, [100 * t[s]["neg_rate"] for s in SOURCES], "o", color=VERM, ms=4, label="Negative-shift rate")
    ax1b.set_ylabel("Ratings with negative shift (%)", color=VERM)
    ax1b.tick_params(axis="y", colors=VERM)
    ax1b.set_ylim(0, 12)
    ax1b.legend(loc="upper left", fontsize=7)
    w = 0.38
    ax2.bar(x - w / 2, [t[s]["aff_rel"] for s in SOURCES], w, color=GREEN, label="Affective/relational index (per 100 words)")
    ax2.bar(x + w / 2, [t[s]["sent_sd"] for s in SOURCES], w, color=GREY, label="Sentence-length SD (words)")
    ax2.set_xticks(x, LABELS, fontsize=7.5)
    ax2.set_title("(b) Text profiles, unique arguments")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.set_ylim(0, 11.5)
    save(fig, "fig1_models")


def fig2():
    t2 = A["table2"]
    prompts = ["Expert Writer Rhetorics", "Compelling Case", "Logical Reasoning", "Deceptive"]
    plab = ["Expert Writer\nRhetorics", "Compelling\nCase", "Logical\nReasoning", "Deceptive"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), gridspec_kw={"hspace": 0.45, "height_ratios": [1, 1.25]})
    x = np.arange(4)
    w = 0.38
    ax1.bar(x - w / 2, [t2[p]["aff_rel"] for p in prompts], w, color=GREEN, label="Affective/relational index")
    ax1.bar(x + w / 2, [t2[p]["analytic"] for p in prompts], w, color=BLUE, label="Analytical index")
    ax1.set_xticks(x, plab, fontsize=7.5)
    ax1.set_ylabel("Per 100 words (unique arguments)")
    ax1.set_title("(a) Prompt type: cue indices and shift")
    ax1.legend(loc="upper center", fontsize=7)
    ax1.set_ylim(0, 8.5)
    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.plot(x, [t2[p]["mean"] for p in prompts], "D-", color=VERM, ms=4, lw=1)
    ax1b.set_ylabel("Mean belief shift", color=VERM)
    ax1b.tick_params(axis="y", colors=VERM)
    ax1b.set_ylim(0.2, 0.55)
    feats = list(FEAT_LABEL)
    y = np.arange(len(feats))[::-1]
    rho = [A["spearman_pooled"][f]["rho"] for f in feats]
    beta = [A["ols_main"]["coef"][f]["beta"] for f in feats]
    se = [A["ols_main"]["coef"][f]["se"] for f in feats]
    ax2.axvline(0, color="black", lw=0.6)
    ax2.plot(rho, y + 0.15, "o", color=GREY, ms=4, label="Pooled Spearman ρ (n = 1,120 arguments)")
    ax2.errorbar(beta, y - 0.15, xerr=1.96 * np.array(se), fmt="s", color=BLUE, ms=4, elinewidth=0.9, capsize=2,
                 label="Clustered OLS β ± 95% CI (n = 3,360 ratings)")
    ax2.set_yticks(y, [FEAT_LABEL[f] for f in feats], fontsize=7)
    ax2.set_xlabel("Standardized association with belief shift")
    ax2.set_title("(b) Pooled vs. adjusted associations")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=7)
    save(fig, "fig2_prompts")


def fig3():
    bi = A["by_init"]
    cv = A["cv"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.45})
    x = np.arange(1, 8)
    w = 0.38
    ax1.bar(x - w / 2, [100 * b["pos"] for b in bi], w, color=GREEN, label="Positive shift (%)")
    ax1.bar(x + w / 2, [100 * b["neg"] for b in bi], w, color=VERM, label="Negative shift (%)")
    ax1.set_xticks(x)
    ax1.set_xlabel("Initial rating (1 = strongly oppose, 7 = strongly support)", fontsize=7.5)
    ax1.set_ylabel("Share of ratings (%)")
    ax1.set_title("(a) Outcomes by initial rating")
    ax1.set_ylim(0, 65)
    ax1.legend(loc="upper right", fontsize=7)
    sets = [("text", "Text only"), ("init", "Initial rating"), ("init_text", "Initial rating\n+ text")]
    xs = np.arange(3)
    for k, (mname, col) in enumerate([("ridge", BLUE), ("hgbt", ORANGE)]):
        vals = [cv[f"{mname}_{s}"]["r2"] for s, _ in sets]
        ax2.bar(xs + (k - 0.5) * w, vals, w, color=col, label={"ridge": "Ridge", "hgbt": "Gradient boosting"}[mname])
        for xi, v in zip(xs + (k - 0.5) * w, vals):
            ax2.text(xi, v + (0.002 if v >= 0 else -0.002), f"{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=6.5)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(xs, [s[1] for s in sets], fontsize=7)
    ax2.set_ylabel("Out-of-fold $R^2$ (claim-grouped 5-fold CV)")
    ax2.set_title("(b) Predicting shift for unseen claims")
    ax2.legend(loc="lower center", fontsize=7)
    ax2.set_ylim(-0.085, 0.075)
    save(fig, "fig3_prediction")


def fig4():
    rp = B["rank_profile"]
    ip = B["intent_profile"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.45})
    ranks = ["1", "2", "3", "4", "5"]
    x = np.arange(5)
    ax1.bar(x, [rp[r]["median_words"] for r in ranks], 0.6, color=SKY)
    ax1.set_xticks(x, [f"Rank {r}" for r in ranks])
    ax1.set_ylabel("Median document length (words)")
    ax1.set_title("(a) Retrieved pages by Google rank")
    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.plot(x, [rp[r]["citations_10k"] for r in ranks], "o-", color=VERM, ms=4, lw=1)
    ax1b.set_ylabel("Citation markers per 10,000 words", color=VERM)
    ax1b.tick_params(axis="y", colors=VERM)
    ax1b.set_ylim(0, 8)
    intents = ["Informational", "Opinion/Debate", "Transactional"]
    ilab = [f"Informational\n(n = {ip['Informational']['queries']})",
            f"Opinion/Debate\n(n = {ip['Opinion/Debate']['queries']})",
            f"Transactional\n(n = {ip['Transactional']['queries']})"]
    feats = [("commercial", "Commercial terms", ORANGE), ("authority", "Authority terms", BLUE),
             ("hedges", "Hedges", GREY)]
    xi = np.arange(3)
    w = 0.26
    for k, (f, lab, col) in enumerate(feats):
        ax2.bar(xi + (k - 1) * w, [ip[i][f] for i in intents], w, color=col, label=lab + " (per 100 words)")
    ax2.set_xticks(xi, ilab, fontsize=7)
    ax2.set_title("(b) Retrieved pages by query intent")
    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ax2b.plot(xi, [100 * ip[i]["inst_share"] for i in intents], "k^", ms=5, label="Institutional/reference share")
    ax2b.set_ylabel("Institutional/reference domains (%)")
    ax2b.set_ylim(0, 35)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.3)
    ax2.set_ylim(0, 0.95)
    save(fig, "fig4_geobench")


def graphical_abstract():
    t, cp = A["table1"], A["claude_pooled"]
    tost = A["tost_args"]
    cv = A["cv"]
    fr = B["rank_friedman"]
    fig = plt.figure(figsize=(12, 6), dpi=200)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, "Lexical Cues in LLM Persuasion and in Web Search Sources", ha="center",
             fontsize=19, weight="bold", color="#1a1a1a")
    fig.text(0.5, 0.885, "Reanalysis of Anthropic/persuasion (3,939 ratings of 1,313 arguments) and GEO-bench "
             "(4,996 pages for 1,000 queries)", ha="center", fontsize=11.5, color="#444")
    panels = [
        ("Human vs. Claude arguments", BLUE, [
            f"Claude 3 Opus {t['Claude 3 Opus']['mean']:.2f} vs. human {t['Human']['mean']:.2f}",
            "mean shift; no detectable difference",
            f"(argument-level p = {A['mw_args']['opus_vs_human']['p']:.2f}), equivalent",
            f"only within ±0.20 (p = {tost['0.2']['p_tost']:.3f})",
            "",
            f"Negative shifts: {100 * cp['neg_k'] / cp['n']:.1f}% of Claude ratings",
            f"vs. {100 * t['Human']['neg_rate']:.1f}% of human ratings (1 of 522),",
            "same claims, same initial attitudes",
        ]),
        ("Do the words predict shift?", ORANGE, [
            "Pooled correlations shrink once",
            "prompt type and initial rating",
            "are controlled",
            "",
            "Unseen claims, text features only:",
            f"R² = {cv['ridge_text']['r2']:.3f} (ridge), {cv['hgbt_text']['r2']:.3f} (boosting)",
            f"Initial rating alone: R² = {cv['ridge_init']['r2']:.3f} to {cv['hgbt_init']['r2']:.3f}",
        ]),
        ("Google top-5 pages (GEO-bench)", GREEN, [
            "Cue rates barely change",
            "across ranks 1 to 5",
            f"(Kendall's W ≤ {max(v['kendall_w'] for v in fr.values()):.3f}); citation",
            "markers are more frequent at",
            "rank 1 than at rank 5",
            "",
            "Pages differ by query intent:",
            "transactional pages carry about 4×",
            "more commercial terms than",
            "informational pages",
        ]),
    ]
    for k, (title, col, lines) in enumerate(panels):
        x0 = 0.035 + k * 0.32
        ax = fig.add_axes([x0, 0.17, 0.295, 0.66])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(col)
            s.set_linewidth(2.5)
        ax.set_facecolor("#fafafa")
        ax.text(0.5, 0.91, title, ha="center", fontsize=12.5, weight="bold", color=col, transform=ax.transAxes)
        for j, line in enumerate(lines):
            ax.text(0.05, 0.77 - j * 0.076, line, fontsize=10.5, transform=ax.transAxes, color="#1a1a1a")
    fig.text(0.5, 0.075, "Word counts explain almost none of the variation in belief change; "
             "prompt type and readers' starting attitudes matter more.", ha="center", fontsize=12.5,
             weight="bold", color="#1a1a1a")
    fig.savefig(FIG / "graphical_abstract.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    graphical_abstract()
    print("figures written to", FIG)
