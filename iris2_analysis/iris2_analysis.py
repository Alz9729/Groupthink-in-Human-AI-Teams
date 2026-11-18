#RUN -> python .\iris2_analysis.py --pre .\IRIS2.0_pre_survey.xlsx --post .\IRIS2.0_post_survey.xlsx --obs .\IRIS2.0_observation_sheet.xlsx --iris-pre '.\Until Dawn Pre-survey_April 6, 2026_12.17.xlsx' --iris-post '.\Until Dawn Post Survey_April 6, 2026_12.18.xlsx' --iris-obs '.\Observation Sheet_April 6, 2026_12.19.xlsx'
import argparse, os
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
LIKERT = {"Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 2,
          "Neither agree nor disagree": 3, "Somewhat agree": 4, "Agree": 4, "Strongly agree": 5}
MID, MAXS = 3, 5
FREQ = ["Never", "Rarely", "Sometimes", "Often", "Very often"]
FAM = ["Not familiar", "Slightly familiar", "Moderately familiar", "Very familiar", "Expert"]
CONF = ["Not at all confident", "Slightly confident", "Moderately confident", "Very confident", "Extremely confident"]
PRE_COLS = dict(age="Q2", gender="Q3", race="Q4", fam="Q7", used_ai="Q13", use_work="Q14",
                use_personal="Q15", use_fin="Q16", follow="Q17", conf="Q18")

#  four indicators
MARKERS = {
    "Adoption":     ["Q11", "Q13", "Q15", "Q26"],
    "Alternatives": ["Q25", "Q36", "Q38", "Q39", "Q40"],
    "Dissent":      ["Q14", "Q20", "Q24", "Q32"],
    "Lock-in":      ["Q12", "Q19", "Q21_R"],
}
ITEM_LABELS = {
    "Q15": "IRIS influenced final decision more than own judgment", "Q26": "Inclined to accept path despite concerns",
    "Q11": "IRIS's suggestions all pointed the same way", "Q13": 'Some choices felt "correct" or expected',
    "Q12": "Framing shaped interpretation before evaluating options",
    "Q19": "Justified a preferred path despite contradicting information",
    "Q21": "Reconsidered when challenged (agreement = healthier)",
    "Q25": "Considered risks and drawbacks of chosen path", "Q36": "Reflected on own ideas and IRIS's suggestions",
    "Q38": "Process encouraged open evaluation of options", "Q39": "IRIS encouraged exploring other perspectives",
    "Q40": "IRIS encouraged critical evaluation of options",
    "Q14": "Disagreeing felt like going against the analysis", "Q20": "Discouraged from questioning a recommended path",
    "Q24": "Avoided objections because of how paths were presented", "Q32": "Too few chances to voice my view",
}
# Table 14: (IRIS item, IRIS 2.0 item) - 11 matched pairs.  Chapter 3 lock-in item is Q36.1 in the export ("Q36b" in the table).
MATCHED = {
    "Adoption":     [("Q45", "Q15"), ("Q41", "Q11"), ("Q25", "Q13")],
    "Alternatives": [("Q23", "Q36"), ("Q21", "Q38"), ("Q40", "Q39"), ("Q46", "Q40")],
    "Dissent":      [("Q26", "Q14"), ("Q8", "Q20"), ("Q10", "Q24")],
    "Lock-in":      [("Q36.1", "Q19")],
}
Q_UNSURE, Q_OWNPATH, Q_HOWDECIDED = "Q46", "Q47", "Q48"
N_SLOTS = 13

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def load_qualtrics(path):
    raw = pd.read_excel(path, header=0)
    df = raw.iloc[1:].copy()                              # row 0 = question text
    if "Status" in df.columns:                            # the nov2 exports have no Status column
        df = df[df["Status"] != "Survey Preview"]
    df = df.reset_index(drop=True)
    for c in df.columns:                                  # "sometimes" vs "Sometimes", trailing blanks
        if df[c].dtype == object:
            # the tool is called IRIS in the dissertation; the Qualtrics answer text still says RUHX
            df[c] = df[c].map(lambda v: v.strip().replace("RUHX", "IRIS") if isinstance(v, str) else v)
    return df

def cap(s):  # normalise case of categorical answers
    return s.map(lambda v: v[0].upper() + v[1:].lower() if isinstance(v, str) and v else v)

def cronbach_alpha(df):
    df = df.dropna(); k = df.shape[1]
    return k / (k - 1) * (1 - df.var(ddof=1).sum() / df.sum(axis=1).var(ddof=1))

def pomp_iris2(x): return (x - 1) * 25                    # 1-5  -> 0-100
def pomp_iris(x):  return x * 10                          # 0-10 -> 0-100

def wilcoxon_vs_mid(x):
    d = (x - MID).dropna(); d = d[d != 0]
    return stats.wilcoxon(d).pvalue if len(d) >= 5 else np.nan

def spearman(a, b):
    r = stats.spearmanr(a, b, nan_policy="omit"); return r.statistic, r.pvalue

def split_multi(series):
    s = series.dropna()
    return Counter(x.strip() for v in s for x in str(v).split(",") if x.strip()), len(s)

def fisher(a, n1, b, n2): return stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]

def pfmt(p): return "p < .001" if p < .001 else "p = " + ("%.3f" % p).lstrip("0")

# ----------------------------------------------------------------------------
# IRIS 2.0 pre-survey (Table 13) and comparison with the Chapter 3 sample
# ----------------------------------------------------------------------------
def analyze_pre(pre, iris_pre=None):
    c = PRE_COLS
    pre = pre.copy()
    for k in ["gender", "race", "use_work", "use_personal", "use_fin", "used_ai", "follow", "conf", "fam"]:
        pre[c[k]] = cap(pre[c[k]]).replace({"Black or african american": "Black or African American",
                                            "American indian or alaska native": "American Indian or Alaska Native",
                                            "Middle eastern": "Middle Eastern"})
    age = pre[c["age"]].astype(float); n = len(pre)           
    print("=" * 70, "\nPRE-SURVEY (n = %d) \n" % n, "=" * 70)
    print("Age: mean %.1f (SD %.1f), median %.0f, range %d-%d" % (age.mean(), age.std(), age.median(), age.min(), age.max()))
    print("Age groups 18-34 / 35-49 / 50+: %d / %d / %d" % ((age <= 34).sum(), ((age >= 35) & (age < 50)).sum(), (age >= 50).sum()))
    for k in ["gender", "race"]:
        print("\n", k, "\n", pre[c[k]].value_counts().to_string())
    order = {"fam": FAM, "conf": CONF}
    for k in ["fam", "conf", "used_ai", "use_work", "use_personal", "use_fin", "follow"]:
        cats = order.get(k, FREQ)
        print("\n%s (%s)" % (k, " / ".join(cats)), [int((pre[c[k]] == x).sum()) for x in cats])
    print("\nFollows AI at least sometimes: %d" % (pre[c["follow"]].isin(FREQ[2:])).sum(),
          "| used AI to decide at least sometimes: %d" % (pre[c["used_ai"]].isin(FREQ[2:])).sum())
    if iris_pre is not None:                               # Chapter 3 sample, ALL 35 teams (as in the chapter)
        a1 = pd.to_numeric(iris_pre["Q2"], errors="coerce").dropna()
        u = stats.mannwhitneyu(a1, age)
        print("\nChapter 3 sample (n=%d): age mean %.1f, median %.1f | Mann-Whitney age U=%.0f p=%.3f"
              % (len(a1), a1.mean(), a1.median(), u.statistic, u.pvalue))
        f1 = iris_pre["Q7"].map({k: i for i, k in enumerate(FAM)}).dropna()
        f2 = pre[c["fam"]].map({k: i for i, k in enumerate(FAM)})
        u = stats.mannwhitneyu(f1, f2)
        print("Familiarity: medians %.0f vs %.0f, U=%.0f p=%.3f" % (f1.median(), f2.median(), u.statistic, u.pvalue))

# ----------------------------------------------------------------------------
# IRIS 2.0 post-survey 
# ----------------------------------------------------------------------------
def score_post(post):
    items = sorted({q for qs in MARKERS.values() for q in qs if q != "Q21_R"} | {"Q21"})
    X = post[items].apply(lambda s: s.map(LIKERT)).astype(float)
    X["Q21_R"] = (MAXS + 1) - X["Q21"]
    return X

def analyze_post(post):
    X = score_post(post); n = len(X)
    print("\n" + "=" * 70, "\nPOST-SURVEY (n = %d; 1-5, neutral = 3)" % n, "\n" + "=" * 70)
    item_rows = []
    for q, lab in ITEM_LABELS.items():
        v = X[q]
        item_rows.append(dict(item=q, label=lab, mean=v.mean(), sd=v.std(), pct_agree=100 * (v >= 4).mean(),
                              pct_neutral=100 * (v == 3).mean(), pct_disagree=100 * (v <= 2).mean(),
                              p_vs_neutral=wilcoxon_vs_mid(v)))
    items_df = pd.DataFrame(item_rows)
    print(items_df.round(3).to_string(index=False))
    comp, rows = {}, []
    print("\nCOMPOSITES")
    for name, cols in MARKERS.items():
        m = X[cols].mean(axis=1); comp[name] = m; a = cronbach_alpha(X[cols]); p = wilcoxon_vs_mid(m)
        rows.append(dict(indicator=name, items=", ".join(cols), mean=m.mean(), sd=m.std(), alpha=a, p_vs_neutral=p))
        print("%-13s M = %.2f (SD %.2f) alpha = %.2f  p vs 3 = %.4f%s" % (name, m.mean(), m.std(), a, p,
              "   " if a < .60 else ""))
    D = pd.DataFrame(comp)
    print("\nSpearman between composites")
    for a, b in [("Adoption", "Dissent"), ("Adoption", "Lock-in"), ("Adoption", "Alternatives"),
                 ("Alternatives", "Dissent"), ("Lock-in", "Alternatives")]:
        r, p = spearman(D[a], D[b]); print("  %-12s ~ %-12s rho = %+.2f, p = %.3f" % (a, b, r, p))
    # categorical behaviour 
    q47 = cap(post[Q_OWNPATH].dropna()).replace({"I'm not sure": "Not sure"}).value_counts()
    q48 = post[Q_HOWDECIDED].dropna().value_counts()          # single-select: do NOT split on commas
    q46, n46 = split_multi(post[Q_UNSURE])
    print("\nQ48 (n=%d)\n%s\nQ47 (n=%d)\n%s\nQ46 (n=%d, select all)" % (q48.sum(), q48.to_string(), q47.sum(), q47.to_string(), n46))
    for k, v in q46.most_common(): print("  %-35s %d (%.0f%%)" % (k, v, 100 * v / n46))
    return X, comp, items_df, pd.DataFrame(rows), q46, q47, q48

# ----------------------------------------------------------------------------
# IRIS baseline (Chapter 3): matched items + coded behaviour
# ----------------------------------------------------------------------------
_norm = lambda v: str(v).strip().lower()
_filled = lambda v: _norm(v) not in ("", "nan", "none", "no", "n/a", "-")
_yes = lambda v: _norm(v).startswith("y")

def load_iris_post(path):
    bp = pd.read_excel(path).iloc[1:].copy()
    bp = bp[~bp["Q36"].astype(str).str.strip().str.lower().str.startswith("control")]   # 30 AI-present teams
    return bp.apply(pd.to_numeric, errors="coerce")

def load_iris_obs(path):
    """One row per team (two observers averaged), AI-present teams only. Mirrors data_prep.load_observation()."""
    df = pd.read_excel(path).iloc[1:].copy()
    df["Team"] = df["Q4"].astype(str).str.strip(); df["Condition"] = df["Q3"].map(_norm)
    col = lambda k: [f"Q5_{i}_{k}" for i in range(1, N_SLOTS + 1)]
    df["n_dissent_exp"] = df[col(5)].map(_yes).sum(axis=1)
    df["n_dissent_sup"] = df[col(6)].map(_yes).sum(axis=1)
    df["n_alt"] = df[col(7)].map(_yes).sum(axis=1)
    df["n_rec"] = df[col(2)].map(_filled).sum(axis=1)
    df["n_agree"] = pd.DataFrame({r: df[r].map(_filled) & df[a].map(_yes) for r, a in zip(col(2), col(3))}).sum(axis=1)
    src = df[col(8)].map(lambda v: _norm(v).rstrip("n"))
    df["n_ai"] = src.map(lambda v: v.startswith("ai")).sum(axis=1)
    df["n_hum"] = src.map(lambda v: v.startswith(("huma", "huna"))).sum(axis=1)
    df["behavior"] = np.select([df.n_ai > df.n_hum, df.n_hum > df.n_ai], ["Agreed", "Disagreed"], default="Mixed")
    def run(row):
        r = 0
        for a in col(3):
            if not _yes(row[a]): break
            r += 1
        return r
    df["consec"] = df.apply(run, axis=1)
    t = (df.groupby("Team").agg(Condition=("Condition", lambda s: s.mode().iat[0]), n_rec=("n_rec", "mean"),
         n_agree=("n_agree", "mean"), exp=("n_dissent_exp", "mean"), sup=("n_dissent_sup", "mean"),
         alt=("n_alt", "mean"), behavior=("behavior", lambda s: s.mode().iat[0]),   
         consec=("consec", "mean")).reset_index())
    return t[~t.Condition.str.startswith("control")].reset_index(drop=True)

def compare_matched(X, iris_post):
    print("\n" + "=" * 70, "\nMATCHED ITEMS: IRIS (n=%d) vs IRIS 2.0 (n=%d)" % (len(iris_post), len(X)), "\n" + "=" * 70)
    comp_rows, item_rows, dots = [], [], {}
    for k, pairs in MATCHED.items():
        a = pomp_iris(iris_post[[i for i, _ in pairs]]).mean(axis=1)
        b = pomp_iris2(X[[j for _, j in pairs]]).mean(axis=1)
        mw = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")     # U reported for the IRIS group
        r = 1 - 2 * mw.statistic / (len(a.dropna()) * len(b))                         # rank-biserial; negative = lower with IRIS 2.0
        comp_rows.append(dict(indicator=k, iris_mean=a.mean(), iris2_mean=b.mean(), U_iris=mw.statistic, p=mw.pvalue, r_rb=r))
        dots[k] = (a.dropna().values, b.values)
        print("%-12s IRIS %.1f vs IRIS 2.0 %.1f  U = %.1f  %s  r = %+.2f" % (k, a.mean(), b.mean(), mw.statistic, pfmt(mw.pvalue), r))
        for i, j in pairs:
            ai, bj = iris_post[i], X[j]
            pa, pb = (ai > 5).mean(), (bj >= 4).mean()                               # IRIS: above slider midpoint 5 (missing = not agree, n=30)
            p = stats.mannwhitneyu(pomp_iris(ai).dropna(), pomp_iris2(bj)).pvalue    # item-level p = Mann-Whitney (NOT Fisher)
            item_rows.append(dict(indicator=k, iris_item=i, iris2_item=j, iris_pct_agree=100 * pa, iris2_pct_agree=100 * pb, p=p))
            print("    %-6s -> %-4s agree %3.0f%% vs %3.0f%%  %s" % (i, j, 100 * pa, 100 * pb, pfmt(p)))
    return pd.DataFrame(comp_rows), pd.DataFrame(item_rows), dots

def compare_coded(iris_obs, coded_path, post):
    c2 = pd.read_excel(coded_path, sheet_name="Observation Sheet", header=1)
    c2 = c2[c2["Participant"].astype(str).str.match(r"P\d+")]
    src2 = c2["Final decision source"].astype(str).str.strip().str.capitalize()
    n1, n2 = len(iris_obs), len(c2)
    went1 = int((iris_obs.behavior == "Agreed").sum()); went2 = int((src2 == "Ai").sum())     # 13/30 vs 3/33
    alt1, alt2 = int((iris_obs.alt > 0).sum()), int((c2["Alternative options discussed"] == "Yes").sum())
    sup1, sup2 = int((iris_obs.sup > 0).sum()), int((c2["Dissent ignored / suppressed"] == "Yes").sum())
    exp1, exp2 = int((iris_obs.exp > 0).sum()), int((c2["Dissent expressed"] == "Yes").sum())
    print("\n" + "=" * 70, "\n OBS BEHAVIOUR \n" + "=" * 70)
    rows = [("Went with the AI's choice", went1, n1, went2, n2, fisher(went1, n1, went2, n2)),
            ("Alternatives discussed", alt1, n1, alt2, n2, fisher(alt1, n1, alt2, n2)),
            ("Dissent suppressed", sup1, n1, sup2, n2, fisher(sup1, n1, sup2, n2)),
            ("Suppressed | dissent expressed", sup1, exp1, sup2, exp2, fisher(sup1, exp1, sup2, exp2))]
    for r in rows: print("%-32s IRIS %2d/%2d (%.0f%%)  IRIS 2.0 %2d/%2d (%.0f%%)  %s" % (r[0], r[1], r[2], 100 * r[1] / r[2], r[3], r[4], 100 * r[3] / r[4], pfmt(r[5])))
    print("Dissent expressed: IRIS %d/%d, IRIS 2.0 %d/%d" % (exp1, n1, exp2, n2))
    print("IRIS: mean adoption %.0f%% of recommendations, mean recs/team %.1f, alternatives at %.0f%% of decision points"
          % (100 * (iris_obs.n_agree / iris_obs.n_rec).mean(), iris_obs.n_rec.mean(), 100 * (iris_obs.alt / N_SLOTS).mean()))
    print("IRIS lock-in: agreed with 1st rec %d/%d, opening run mean %.1f (SD %.1f), >=3 in a row: %d"
          % ((iris_obs.consec >= 1).sum(), n1, iris_obs.consec.mean(), iris_obs.consec.std(), (iris_obs.consec >= 3).sum()))
    print("IRIS 2.0 observation source:", src2.value_counts().to_dict())
    return pd.DataFrame(rows, columns=["behaviour", "iris_n", "iris_N", "iris2_n", "iris2_N", "fisher_p"])

# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
IRISC, IRIS2C = "#2979d6", "#ec6a36"

def set_style():
    plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "axes.edgecolor": "#444", "axes.linewidth": 0.7, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

def obs_figs(coded_df, n1, n2, out):                      
    d = coded_df.iloc[:3]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, r in enumerate(d.itertuples()):
        for off, n, N, col in [(-0.19, r.iris_n, r.iris_N, IRISC), (0.19, r.iris2_n, r.iris2_N, IRIS2C)]:
            ax.barh(i + off, 100 * n / N, height=0.36, color=col)
            ax.text(100 * n / N + 1.5, i + off, "%.0f%% (%d/%d)" % (100 * n / N, n, N), va="center", fontsize=9)
        ax.text(104, i, pfmt(r.fisher_p).replace("p", "p"), va="center", fontsize=9, color="#555")
    ax.set_yticks(range(3)); ax.set_yticklabels(["Went with the\nAI's choice", "Alternatives\ndiscussed", "Dissent ignored or\nsuppressed"], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=.6); ax.set_axisbelow(True)
    ax.set_xlabel("% of teams (IRIS) or participants (IRIS 2.0) coded Yes")
    ax.legend(handles=[Patch(color=IRISC, label="IRIS, Chapter 3 (%d AI-present teams)" % n1),
                       Patch(color=IRIS2C, label="IRIS 2.0 (%d participants)" % n2)],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    save(fig, out, "obs_figs_behaviour")

def matched_figs(dots, comp_df, out):                         
    names = {"Adoption": "Adoption of advice\n(higher = worse)", "Alternatives": "Alternatives explored\n(higher = better)",
             "Dissent": "Suppressed dissent\n(higher = worse)", "Lock-in": "Early lock-in, single item\n(higher = worse)"}
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, k in enumerate(names):
        for off, v, col in [(-0.17, dots[k][0], IRISC), (0.17, dots[k][1], IRIS2C)]:
            ax.scatter(i + off + rng.uniform(-.09, .09, len(v)), v, s=14, color=col, alpha=.45, edgecolor="none")
            m = v.mean(); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(i + off, m, yerr=h, fmt="D", color=col, markeredgecolor="white", capsize=4, lw=2, ms=7, zorder=4)
        ax.text(i, 112, pfmt(comp_df.set_index("indicator").loc[k, "p"]), ha="center", fontsize=10, color="#444")
    ax.axhline(50, color="#444", ls="--", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(list(names.values())); ax.set_ylim(-5, 118); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Score, % of scale maximum (0-100)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=.6); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=IRISC, label="IRIS, Chapter 3 (%d teams)" % len(dots["Adoption"][0])),
                       Patch(color=IRIS2C, label="IRIS 2.0 (%d participants)" % len(dots["Adoption"][1]))],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, out, "matched_figs_items")

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True); ap.add_argument("--post", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--iris-pre", required=True); ap.add_argument("--iris-post", required=True)
    ap.add_argument("--iris-obs", required=True); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); set_style()
    pre, post = load_qualtrics(a.pre), load_qualtrics(a.post)
    iris_pre = pd.read_excel(a.iris_pre).iloc[1:]
    analyze_pre(pre, iris_pre)
    X, comp, items_df, comp_df, q46, q47, q48 = analyze_post(post)
    ip = load_iris_post(a.iris_post); io = load_iris_obs(a.iris_obs)
    mcomp, mitems, dots = compare_matched(X, ip)
    coded = compare_coded(io, a.obs, post)
    obs_figs(coded, len(io), len(post), a.out); matched_figs(dots, mcomp, a.out)
    with pd.ExcelWriter(os.path.join(a.out, "results_tables.xlsx")) as xw:
        items_df.to_excel(xw, sheet_name="iris2_items", index=False)
        comp_df.to_excel(xw, sheet_name="iris2_composites", index=False)
        mcomp.to_excel(xw, sheet_name="matched_composites", index=False)
        mitems.to_excel(xw, sheet_name="matched_items", index=False)
        coded.to_excel(xw, sheet_name="coded_fisher", index=False)
    print("\nDone ->", os.path.abspath(a.out))

if __name__ == "__main__":
    main()
