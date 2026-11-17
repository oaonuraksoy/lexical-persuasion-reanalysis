"""LaTeX tables 1-4 generated from results/results.json (written to tables/)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
A, B = R["corpus_a"], R["corpus_b"]
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)


def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def table1():
    rows = []
    for s in ["Control", "Claude 1.3", "Claude Instant 1.2", "Claude 2", "Claude 3 Haiku", "Claude 3 Opus", "Human"]:
        t = A["table1"][s]
        rows.append(f"{s} & {t['trials']} / {t['args']} & {f(t['mean'],3)} [{f(t['mean_ci'][0])}, {f(t['mean_ci'][1])}] & "
                    f"{t['neg_k']} ({pct(t['neg_rate'])}\\%) [{pct(t['neg_wilson'][0])}, {pct(t['neg_wilson'][1])}] & "
                    f"{t['args_with_neg']} & {f(t['aff_rel'])} & {f(t['analytic'])} & {f(t['sent_sd'])} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{tabular}}{{lrrrrrrr}}
\toprule
Source & Ratings / args & Mean shift [95\% CI] & Negative shifts, \% [Wilson 95\% CI] & Args with $\geq$1 neg. & Aff./rel. & Analytic & Sent.\ SD \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""


def table2():
    rows = []
    for p in ["Control Prompt", "Expert Writer Rhetorics", "Compelling Case", "Logical Reasoning", "Deceptive"]:
        t = A["table2"][p]
        rows.append(f"{p} & {t['trials']} / {t['args']} & {f(t['mean'],3)} & {pct(t['neg_rate'])} & {f(t['collective'])} & "
                    f"{f(t['pronouns'])} & {f(t['empathy'])} & {f(t['connectives'])} & {f(t['evidential'])} & "
                    f"{f(t['numeric'])} & {f(t['flesch'],1)} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{tabular}}{{lrrrrrrrrrr}}
\toprule
Prompt type & Ratings / args & Shift & Neg.\ \% & Collect. & Pron. & Affect. & Connect. & Evid. & Numeric & Flesch \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""


LAB = {"collective": "Collective-reference terms", "connectives": "Logical connectives",
       "evidential": "Evidential nouns", "numeric": "Numeric tokens", "hedges": "Hedges and qualifiers",
       "empathy": "Affective terms", "loss": "Loss/threat terms", "certainty": "Certainty terms",
       "n_words": "Argument length (words)", "sent_sd": "Sentence-length SD"}


def pfmt(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def table3():
    rows = []
    for k, lab in LAB.items():
        pooled = A["spearman_pooled"][k]
        w = [A["spearman_within"][p][k]["rho"] for p in A["spearman_within"]]
        o = A["ols_main"]["coef"][k]
        fe = A["ols_claim_fe"]["coef"][k]
        rows.append(f"{lab} & {f(pooled['rho'],3)} & {pfmt(pooled['q'])} & [{f(min(w),3)}, {f(max(w),3)}] & "
                    f"{f(o['beta'],3)} ({f(o['se'],3)}) & {pfmt(o['p'])} & {f(fe['beta'],3)} & {pfmt(fe['p'])} \\\\")
    o = A["ols_main"]["coef"]["init"]
    fe = A["ols_claim_fe"]["coef"]["init"]
    rows.append(f"Initial rating & -- & -- & -- & {f(o['beta'],3)} ({f(o['se'],3)}) & {pfmt(o['p'])} & {f(fe['beta'],3)} & {pfmt(fe['p'])} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{tabular}}{{lrrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Pooled (1,120 args)}} & Within prompt & \multicolumn{{2}}{{c}}{{OLS, prompt FE}} & \multicolumn{{2}}{{c}}{{OLS, prompt + claim FE}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}
Feature & $\rho$ & $q$ & $\rho$ range & $\beta$ (SE) & $p$ & $\beta$ & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""


def table4():
    rp, ip, fr = B["rank_profile"], B["intent_profile"], B["rank_friedman"]
    rows = []
    for k in ["1", "2", "3", "4", "5"]:
        t = rp[k]
        rows.append(f"Rank {k} & {t['docs']} & {t['median_words']:.1f} & {f(t['commercial'],3)} & {f(t['authority'],3)} & "
                    f"{f(t['hedges'],3)} & {f(t['numeric'])} & {f(t['citations_10k'])} & {pct(t['inst_share'],1)} \\\\")
    rows.append(r"\midrule")
    rows.append("Friedman $q$ (Kendall's $W$) & -- & " + " & ".join(
        f"{fr[x]['q']:.2f} ({fr[x]['kendall_w']:.3f})" for x in
        ["n_words", "commercial", "authority", "hedges", "numeric", "citations_10k", "inst_share"]) + r" \\")
    rows.append(r"\midrule")
    for it in ["Informational", "Opinion/Debate", "Transactional"]:
        t = ip[it]
        rows.append(f"{it} ({t['queries']} queries) & {t['docs']} & {t['median_words']:.1f} & {f(t['commercial'],3)} & "
                    f"{f(t['authority'],3)} & {f(t['hedges'],3)} & {f(t['numeric'])} & {f(t['citations_10k'])} & "
                    f"{pct(t['inst_share'],1)} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Stratum & Pages & Median words & Commercial & Authority & Hedges & Numeric & Citations / 10k & Inst./ref.\ \% \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""


if __name__ == "__main__":
    for name, fn in [("table1", table1), ("table2", table2), ("table3", table3), ("table4", table4)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
    print("tables written to", OUT)
