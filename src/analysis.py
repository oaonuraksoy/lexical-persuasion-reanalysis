"""Reanalysis of Anthropic/persuasion (Corpus A) and GEO-bench test (Corpus B).

Usage:  python src/analysis.py            (reads data/*.parquet, writes results/)
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from features import AFF_REL, ANALYTIC, check_disjoint, text_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
SEED = 20261117
N_BOOT = 10_000
MODELS = ["Claude 1.3", "Claude Instant 1.2", "Claude 2", "Claude 3 Haiku", "Claude 3 Opus"]
PROMPTS = ["Expert Writer Rhetorics", "Compelling Case", "Logical Reasoning", "Deceptive"]
TEXT_FEATS = ["collective", "connectives", "evidential", "numeric", "hedges",
              "empathy", "loss", "certainty", "n_words", "sent_sd"]
ALL_TEXT = TEXT_FEATS + ["authority", "commercial", "citations", "questions", "guiraud", "flesch"]


# --------------------------------------------------------------------------- helpers
def wilson(k: int, n: int) -> list[float]:
    lo, hi = proportion_confint(k, n, method="wilson")
    return [float(lo), float(hi)]


def cliff_from_u(u: float, n1: int, n2: int) -> float:
    return 2.0 * u / (n1 * n2) - 1.0


def mwu(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"U": float(r.statistic), "p": float(r.pvalue), "delta": cliff_from_u(r.statistic, len(x), len(y)),
            "n1": len(x), "n2": len(y), "mean1": float(x.mean()), "mean2": float(y.mean())}


def bh(ps) -> list[float]:
    return [float(q) for q in multipletests(ps, method="fdr_bh")[1]]


def boot_mean_ci(values, rng, n_boot=N_BOOT) -> list[float]:
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def welch_tost(x, y, margin: float) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    p = max(stats.t.sf((d + margin) / se, df), stats.t.sf((margin - d) / se, df))
    tq = stats.t.ppf(0.95, df)
    return {"diff": float(d), "se": float(se), "df": float(df), "ci90": [float(d - tq * se), float(d + tq * se)],
            "margin": margin, "p_tost": float(p), "p_welch": float(2 * stats.t.sf(abs(d) / se, df))}


def paired_tost(d, margin: float) -> dict:
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p = max(stats.t.sf((m + margin) / se, n - 1), stats.t.sf((margin - m) / se, n - 1))
    tq = stats.t.ppf(0.95, n - 1)
    return {"diff": float(m), "se": float(se), "n": n, "ci90": [float(m - tq * se), float(m + tq * se)],
            "margin": margin, "p_tost": float(p)}


# --------------------------------------------------------------------------- Corpus A
def load_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_parquet(DATA / "persuasion.parquet")
    a["init"] = a.rating_initial.str[0].astype(int)
    a["final"] = a.rating_final.str[0].astype(int)
    a["y"] = a.persuasiveness_metric.astype(float)
    a["neg"] = (a.y < 0).astype(int)
    a["pos"] = (a.y > 0).astype(int)
    a["is_claude"] = a.source.str.startswith("Claude")
    a["arg_id"] = pd.factorize(a.argument)[0]
    feats = pd.DataFrame([text_features(t) for t in a.argument.drop_duplicates()],
                         index=a.argument.drop_duplicates().values)
    feats["aff_rel"] = feats[AFF_REL].sum(axis=1)
    feats["analytic"] = feats[ANALYTIC].sum(axis=1)
    a = a.join(feats, on="argument")
    g = a.groupby("arg_id")
    args = g.agg(source=("source", "first"), prompt_type=("prompt_type", "first"), claim=("claim", "first"),
                 y=("y", "mean"), neg=("neg", "sum"), n=("y", "size"), init=("init", "mean"))
    args = args.join(g[list(feats.columns)].first())
    return a, args


def corpus_a(rng) -> dict:
    a, args = load_a()
    R: dict = {"n_trials": int(len(a)), "n_args": int(len(args)), "n_workers": int(a.worker_id.nunique()),
               "n_claims": int(a.claim.nunique()),
               "ratings_per_arg": sorted(set(args.n.astype(int).tolist()))}
    claude = a[a.is_claude]
    human = a[a.source == "Human"]

    # ---- Table 1: per-source outcomes and argument-level text profiles
    t1 = {}
    for s in ["Control"] + MODELS + ["Human"]:
        t, ar = a[a.source == s], args[args.source == s]
        k, n = int(t.neg.sum()), len(t)
        t1[s] = {"trials": n, "args": len(ar), "mean": float(t.y.mean()), "mean_ci": boot_mean_ci(ar.y, rng),
                 "neg_k": k, "neg_rate": k / n, "neg_wilson": wilson(k, n),
                 "neg_ci_cluster": boot_mean_ci(ar.neg / ar.n, rng),
                 "args_with_neg": int((ar.neg > 0).sum()),
                 "aff_rel": float(ar.aff_rel.mean()), "analytic": float(ar.analytic.mean()),
                 "sent_sd": float(ar.sent_sd.mean())}
    R["table1"] = t1
    ca = args[args.source.str.startswith("Claude")]
    ha = args[args.source == "Human"]
    R["claude_pooled"] = {"mean": float(claude.y.mean()), "mean_ci": boot_mean_ci(ca.y, rng),
                          "neg_k": int(claude.neg.sum()), "n": len(claude),
                          "neg_wilson": wilson(int(claude.neg.sum()), len(claude)),
                          "neg_ci_cluster": boot_mean_ci(ca.neg / ca.n, rng),
                          "args_with_neg": int((ca.neg > 0).sum()), "n_args": len(ca)}

    # ---- argument-level comparisons of mean shift
    oa = args[args.source == "Claude 3 Opus"]
    c13 = args[args.source == "Claude 1.3"]
    R["kw_models_args"] = dict(zip(["H", "p"], map(float, stats.kruskal(*[args[args.source == m].y for m in MODELS]))))
    raw = {"opus_vs_human": mwu(oa.y, ha.y), "opus_vs_c13": mwu(oa.y, c13.y), "claude_vs_human": mwu(ca.y, ha.y)}
    qs = bh([v["p"] for v in raw.values()])
    for (k, v), q in zip(raw.items(), qs):
        v["q"] = q
    R["mw_args"] = raw

    # equivalence: argument-level Welch TOST and claim-paired TOST
    R["tost_args"] = {str(m): welch_tost(oa.y, ha.y, m) for m in (0.10, 0.15, 0.20)}
    cm = args[args.source.isin(["Claude 3 Opus", "Human"])].groupby(["claim", "source"]).y.mean().unstack()
    d = (cm["Claude 3 Opus"] - cm["Human"]).dropna()
    R["tost_claim_paired"] = {str(m): paired_tost(d, m) for m in (0.10, 0.15, 0.20)}
    # for reference: trial-level TOST as in the earlier version (treats repeated ratings as independent)
    R["tost_trials_reference"] = welch_tost(a[a.source == "Claude 3 Opus"].y, human.y, 0.15)

    # ---- negative shifts
    ka, kh = int((ca.neg > 0).sum()), int((ha.neg > 0).sum())
    R["neg_args_fisher"] = {"claude": [ka, len(ca)], "human": [kh, len(ha)],
                            "claude_rate": ka / len(ca), "human_rate": kh / len(ha),
                            "claude_wilson": wilson(ka, len(ca)), "human_wilson": wilson(kh, len(ha)),
                            "p": float(stats.fisher_exact([[ka, len(ca) - ka], [kh, len(ha) - kh]])[1])}
    R["neg_trials_fisher_reference"] = float(stats.fisher_exact(
        [[int(claude.neg.sum()), len(claude) - int(claude.neg.sum())],
         [int(human.neg.sum()), len(human) - int(human.neg.sum())]])[1])
    ks = stats.ks_2samp(human.init, claude.init)
    R["init_balance"] = {"human_mean": float(human.init.mean()), "claude_mean": float(claude.init.mean()),
                         "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
                         "mw_p": float(stats.mannwhitneyu(human.init, claude.init).pvalue)}
    strat = {}
    for lab, cond in [("init_le4", lambda x: x.init <= 4), ("init_ge5", lambda x: x.init >= 5)]:
        strat[lab] = {"human": [int(human[cond(human)].neg.sum()), int(cond(human).sum())],
                      "claude": [int(claude[cond(claude)].neg.sum()), int(cond(claude).sum())]}
    R["neg_by_init"] = strat
    by_init = []
    for i in range(1, 8):
        t = a[(a.init == i) & (a.source != "Control")]
        by_init.append({"init": i, "n": len(t), "mean": float(t.y.mean()), "pos": float(t.pos.mean()),
                        "neg": float(t.neg.mean())})
    R["by_init"] = by_init

    # ---- deduplicated lexical comparison, Claude vs Human
    lex = {}
    feats = ALL_TEXT + ["pronouns", "aff_rel", "analytic"]
    for f in feats:
        lex[f] = mwu(ca[f], ha[f])
    for f, q in zip(feats, bh([lex[f]["p"] for f in feats])):
        lex[f]["q"] = q
    R["lex_claude_vs_human"] = lex

    # ---- prompt strategies
    ap = args[args.prompt_type.isin(PROMPTS)]
    R["kw_prompt_args"] = dict(zip(["H", "p"], map(float, stats.kruskal(*[ap[ap.prompt_type == p].y for p in PROMPTS]))))
    t2 = {}
    for p in ["Control Prompt"] + PROMPTS:
        t, ar = a[a.prompt_type == p], args[args.prompt_type == p]
        t2[p] = {"trials": len(t), "args": len(ar), "mean": float(t.y.mean()), "neg_rate": float(t.neg.mean()),
                 **{f: float(ar[f].mean()) for f in ["collective", "pronouns", "empathy", "connectives", "evidential",
                                                     "numeric", "loss", "flesch", "aff_rel", "analytic"]}}
    R["table2"] = t2
    dec, exp = ap[ap.prompt_type == "Deceptive"], ap[ap.prompt_type == "Expert Writer Rhetorics"]
    pairs = {}
    for f in ["y", "collective", "numeric", "evidential", "loss"]:
        pairs[f] = mwu(dec[f], exp[f])
    for f, q in zip(pairs, bh([v["p"] for v in pairs.values()])):
        pairs[f]["q"] = q
    R["deceptive_vs_expert"] = pairs

    # ---- correlations: pooled and within prompt (unique Claude arguments)
    pooled = {}
    for f in TEXT_FEATS:
        r = stats.spearmanr(ap[f], ap.y)
        pooled[f] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    for f, q in zip(TEXT_FEATS, bh([pooled[f]["p"] for f in TEXT_FEATS])):
        pooled[f]["q"] = q
    within = {}
    for p in PROMPTS:
        s = ap[ap.prompt_type == p]
        res = {f: stats.spearmanr(s[f], s.y) for f in TEXT_FEATS}
        qs = bh([res[f].pvalue for f in TEXT_FEATS])
        within[p] = {f: {"rho": float(res[f].statistic), "p": float(res[f].pvalue), "q": q}
                     for f, q in zip(TEXT_FEATS, qs)}
    R["spearman_pooled"] = pooled
    R["spearman_within"] = within

    # ---- clustered OLS on Claude trials (z-scored predictors, prompt fixed effects)
    c = claude.copy()
    zcols = TEXT_FEATS + ["init", "pronouns"]
    for f in zcols:
        c["z_" + f] = (c[f] - c[f].mean()) / c[f].std(ddof=0)

    def ols(cols, extra=""):
        form = "y ~ " + " + ".join("z_" + f for f in cols) + " + z_init + C(prompt_type)" + extra
        m = smf.ols(form, data=c).fit(cov_type="cluster", cov_kwds={"groups": c.arg_id})
        return {f: {"beta": float(m.params["z_" + f]), "se": float(m.bse["z_" + f]), "p": float(m.pvalues["z_" + f])}
                for f in cols + ["init"]}, float(m.rsquared), int(m.nobs)

    main, r2, nobs = ols(TEXT_FEATS)
    R["ols_main"] = {"coef": main, "r2": r2, "n": nobs, "clusters": int(c.arg_id.nunique())}
    pron_cols = ["pronouns" if f == "collective" else f for f in TEXT_FEATS]
    R["ols_pronouns"] = {"coef": ols(pron_cols)[0]}
    R["ols_claim_fe"] = {"coef": ols(TEXT_FEATS, " + C(claim)")[0]}

    # ---- claim-disjoint 5-fold cross-validation (Claude + Human trials)
    cv = a[a.source != "Control"].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    sets = {"text": ALL_TEXT, "init": ["init"], "init_text": ["init"] + ALL_TEXT}
    regs = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=50.0)),
            "hgbt": lambda: HistGradientBoostingRegressor(random_state=0)}
    clfs = {"ridge": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "hgbt": lambda: HistGradientBoostingClassifier(random_state=0)}
    cvres = {}
    for sname, cols in sets.items():
        X = cv[cols].to_numpy(float)
        for mname in regs:
            pred = np.zeros(len(cv))
            prob = np.zeros(len(cv))
            for tr, te in gkf.split(X, cv.y, cv.claim):
                pred[te] = regs[mname]().fit(X[tr], cv.y.values[tr]).predict(X[te])
                prob[te] = clfs[mname]().fit(X[tr], cv.pos.values[tr]).predict_proba(X[te])[:, 1]
            cvres[f"{mname}_{sname}"] = {"r2": float(r2_score(cv.y, pred)),
                                         "mae": float(np.mean(np.abs(cv.y - pred))),
                                         "auroc": float(roc_auc_score(cv.pos, prob))}
    # argument-level (deduplicated) text-only fit
    av = args[args.source != "Control"].reset_index(drop=True)
    X = av[ALL_TEXT].to_numpy(float)
    for mname in regs:
        pred = np.zeros(len(av))
        for tr, te in gkf.split(X, av.y, av.claim):
            pred[te] = regs[mname]().fit(X[tr], av.y.values[tr]).predict(X[te])
        cvres[f"{mname}_text_args"] = {"r2": float(r2_score(av.y, pred)), "n": len(av)}
    R["cv"] = cvres
    R["cv_n"] = {"trials": len(cv), "claims": int(cv.claim.nunique())}
    r = stats.pearsonr(cv.init, cv.y)
    R["init_insample"] = {"r": float(r.statistic), "r2": float(r.statistic ** 2), "p": float(r.pvalue)}
    return R, a, args


# --------------------------------------------------------------------------- Corpus B
INSTITUTIONAL_TLD = (".gov", ".edu", ".int", ".mil")
ENCYCLOPEDIC = ("wikipedia.org", "britannica.com", "encyclopedia.com", "newworldencyclopedia.org", "wikibooks.org",
                "merriam-webster.com", "dictionary.com", "cambridge.org/dictionary", "collinsdictionary.com")
SCHOLARLY = ("frontiersin.org", "nature.com", "sciencedirect.com", "springer.com", "link.springer.com", "wiley.com",
             "tandfonline.com", "mdpi.com", "plos.org", "jstor.org", "arxiv.org", "researchgate.net",
             "academic.oup.com", "sagepub.com", "bmj.com", "thelancet.com", "cell.com", "science.org",
             "pnas.org", "ieee.org", "acm.org", "biomedcentral.com", "semanticscholar.org", "doi.org", "ssrn.com")
ECOMMERCE = ("amazon.", "ebay.", "walmart.com", "etsy.com", "bestbuy.com", "target.com", "aliexpress.com",
             "alibaba.com", "homedepot.com", "lowes.com", "costco.com", "ikea.com", "wayfair.com", "newegg.com",
             "flipkart.com", "shopify.com", "zappos.com", "macys.com", "kohls.com", "overstock.com")


def domain_class(url: str) -> str:
    h = urlparse(url).netloc.lower().removeprefix("www.")
    if any(h.endswith(t) for t in INSTITUTIONAL_TLD) or ".ac." in h or h.endswith(".ac") or ".gov." in h or ".edu." in h:
        return "institutional"
    if any(h.endswith(e) or e in h for e in ENCYCLOPEDIC):
        return "encyclopedic"
    if any(h.endswith(s) for s in SCHOLARLY):
        return "scholarly"
    if any(h.startswith(e) or ("." + e) in h for e in ECOMMERCE):
        return "ecommerce"
    return "general"


def intent_of(tags) -> str:
    t = {x.lower() for x in tags}
    if "transactional" in t:
        return "Transactional"
    if t & {"opinion", "debate"}:
        return "Opinion/Debate"
    return "Informational"


B_FEATS = ["n_words", "numeric", "authority", "commercial", "hedges", "citations_10k"]


def corpus_b() -> tuple[dict, pd.DataFrame]:
    b = pd.read_parquet(DATA / "geobench_test.parquet")
    rows = []
    for qi, r in b.iterrows():
        it = intent_of(r.tags)
        for k, s in enumerate(r.sources):
            f = text_features(s["cleaned_text"])
            rows.append({"q": qi, "rank": k + 1, "intent": it, "url": s["url"], "domain": domain_class(s["url"]), **f})
    d = pd.DataFrame(rows)
    d["citations_10k"] = d.citations * 100
    d["inst_share"] = d.domain.isin(["institutional", "encyclopedic", "scholarly"]).astype(float)
    R: dict = {"n_docs": len(d), "n_queries": int(d.q.nunique()),
               "docs_by_rank": d["rank"].value_counts().sort_index().astype(int).to_dict(),
               "queries_by_intent": b.tags.map(intent_of).value_counts().to_dict(),
               "docs_by_intent": d.intent.value_counts().to_dict(),
               "domain_counts": d.domain.value_counts().to_dict()}
    # rank profiles (descriptive) and within-query Friedman tests
    prof = {}
    for k in range(1, 6):
        s = d[d["rank"] == k]
        prof[k] = {"docs": len(s), "median_words": float(s.n_words.median()),
                   **{f: float(s[f].mean()) for f in B_FEATS[1:]}, "inst_share": float(s.inst_share.mean())}
    R["rank_profile"] = prof
    full = d.groupby("q").filter(lambda g: len(g) == 5)
    fr = {}
    for f in B_FEATS + ["inst_share"]:
        w = full.pivot(index="q", columns="rank", values=f).dropna()
        st = stats.friedmanchisquare(*[w[c] for c in w.columns])
        n, k = len(w), w.shape[1]
        fr[f] = {"chi2": float(st.statistic), "p": float(st.pvalue), "n": n, "kendall_w": float(st.statistic / (n * (k - 1)))}
    for f, q in zip(fr, bh([v["p"] for v in fr.values()])):
        fr[f]["q"] = q
    R["rank_friedman"] = fr
    # rank 1 vs rank 5, paired within query
    r15 = {}
    for f in B_FEATS:
        w = full.pivot(index="q", columns="rank", values=f).dropna()
        r15[f] = {"p": float(stats.wilcoxon(w[1], w[5]).pvalue), "mean1": float(w[1].mean()), "mean5": float(w[5].mean())}
    for f, q in zip(r15, bh([v["p"] for v in r15.values()])):
        r15[f]["q"] = q
    R["rank1_vs_5"] = r15
    # intent profiles (document-level descriptives) and query-level tests
    ip = {}
    for it in ["Informational", "Opinion/Debate", "Transactional"]:
        s = d[d.intent == it]
        ip[it] = {"docs": len(s), "queries": int(s.q.nunique()), "median_words": float(s.n_words.median()),
                  **{f: float(s[f].mean()) for f in B_FEATS[1:]}, "inst_share": float(s.inst_share.mean()),
                  "ecommerce_share": float((s.domain == "ecommerce").mean())}
    R["intent_profile"] = ip
    qm = d.groupby(["q", "intent"])[B_FEATS + ["inst_share"]].mean().reset_index()
    tests = {}
    for f in ["commercial", "authority", "hedges", "inst_share"]:
        for other in ["Transactional", "Opinion/Debate"]:
            tests[f"{f}:{other}_vs_Informational"] = mwu(qm[qm.intent == other][f], qm[qm.intent == "Informational"][f])
    for k, q in zip(tests, bh([v["p"] for v in tests.values()])):
        tests[k]["q"] = q
    R["intent_tests_querylevel"] = tests
    return R, d


def main() -> None:
    check_disjoint()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    ra, a, args = corpus_a(rng)
    rb, d = corpus_b()
    res = {"corpus_a": ra, "corpus_b": rb, "seed": SEED, "n_boot": N_BOOT}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    args.drop(columns=[]).to_csv(OUT / "corpus_a_arguments.csv")
    d.drop(columns=["url"]).to_csv(OUT / "corpus_b_documents.csv", index=False)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
