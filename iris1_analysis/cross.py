
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr
import warnings
warnings.filterwarnings("ignore")

# File paths 
OBS_FILE  = "Observation Sheet_April 6, 2026_12.19.xlsx"
PRE_FILE  = "Until Dawn Pre-survey_April 6, 2026_12.17.xlsx"
POST_FILE = "Until Dawn Post Survey_April 6, 2026_12.18.xlsx"

output_dir = 'crossfile_survey_figs'
os.makedirs(output_dir, exist_ok=True)

# 1. Load data
pre  = pd.read_excel(PRE_FILE,  header=1)
post = pd.read_excel(POST_FILE, header=1)
obs  = pd.read_excel(OBS_FILE,  header=1)

for df in [pre, post, obs]:
    df["Team ID"] = pd.to_numeric(df["Team ID"], errors="coerce")

# 1b. Normalise condition
def norm_cond(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s == "control": return "Control"
    parts = s.split()
    framing_map = {"open": "Open", "closed": "Closed", "neutral": "Neutral"}
    valence_map = {"positive": "Positive", "negative": "Negative"}
    f   = framing_map.get(parts[0], parts[0].title()) if len(parts) > 0 else ""
    val = valence_map.get(parts[1], parts[1].title()) if len(parts) > 1 else ""
    return f"{f} {val}".strip()

# Use pre-survey as the authoritative condition source (one row = one participant)
pre["condition_norm"] = pre["condition"].apply(norm_cond)
# Also add framing and valence factors
def get_framing(c):
    if pd.isna(c) or c == "Control": return "Control"
    return c.split()[0]   # Open / Closed / Neutral

def get_valence(c):
    if pd.isna(c) or c == "Control": return "Control"
    parts = c.split()
    return parts[1] if len(parts) > 1 else "Unknown"

pre["framing"] = pre["condition_norm"].apply(get_framing)
pre["valence"] = pre["condition_norm"].apply(get_valence)

# 2. Build observation metrics─────────────────────────────────────────────
agree_cols      = [c for c in obs.columns if "Agreement with AI"      in c]
disagree_cols   = [c for c in obs.columns if "Disagreement with AI"   in c]
source_cols     = [c for c in obs.columns if "Final decision source"  in c]
dissent_cols    = [c for c in obs.columns if "Dissent Expressed"      in c]
suppressed_cols = [c for c in obs.columns if "ignored/suppressed"     in c]

def yn(val):
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ["yes","yea","y"]: return 1
        if v in ["no","n"]:        return 0
    return np.nan

for col in agree_cols + disagree_cols + dissent_cols + suppressed_cols:
    obs[col] = obs[col].apply(yn)

obs["total_agree"]      = obs[agree_cols].sum(axis=1)
obs["total_dissent"]    = obs[dissent_cols].sum(axis=1)
obs["total_suppressed"] = obs[suppressed_cols].sum(axis=1)
obs["n_decisions"]      = obs[agree_cols].notna().sum(axis=1)
obs["ai_driven"]        = obs.apply(
    lambda r: sum(1 for c in source_cols
                  if pd.notna(r[c]) and str(r[c]).strip().upper() in ["AI","A.I."]),
    axis=1)

obs_g = obs.groupby("Team ID")[
    ["total_agree","total_dissent","total_suppressed","n_decisions","ai_driven"]
].mean().reset_index()
obs_g["ai_compliance_rate"] = obs_g["total_agree"]      / obs_g["n_decisions"]
obs_g["dissent_rate"]       = obs_g["total_dissent"]    / obs_g["n_decisions"]
obs_g["suppression_rate"]   = obs_g["total_suppressed"] / obs_g["n_decisions"]

def parse_alive(text):
    if pd.isna(text): return np.nan
    t = str(text).lower()
    for n in ["4","3","2","1","0"]:
        if n+" person" in t or n+" people" in t: return int(n)
    if "all" in t and "saved" in t: return 4
    return np.nan

obs["alive_parsed"] = obs["No of people alive"].apply(parse_alive)
alive_g = obs.groupby("Team ID")["alive_parsed"].mean().reset_index()
alive_g.columns = ["Team ID","survivors"]

# 3. Build post-survey composites─────────────────────────────────────────
GT_COLS = [
    "Disagreeing with the AI felt like going against the group direction.",
    "I felt discouraged from questioning or disagreeing during decision-making.",
    "Once an option seemed preferred, I was more likely to justify it even when new information contradicted it.",
    "The AI discouraged us from discussing information that contradicted its recommendations.",
    "I avoided raising objections because of how the AI presented its recommendations.",
    "I avoided raising objections during decion making.",
    "There appeared to be full agreement on the final choice, even if not all alternatives were thoroughly discussed."
]
AI_COLS = [
    "The way the AI framed the situation made certain options seem less relevant or unnecessary to discuss.",
    "The AI made certain choices feel like the \u2018correct\u2019 or expected ones.",
    "The AI influenced our decisions more than my own initial judgment.",
    "When the AI supported a particular option, we were more likely to justify our earlier decision even when new information contradicted it.",
    "The AI discouraged us from discussing information that contradicted its recommendations."
]
SAT_COLS = [
    "I was confident that the decision-making process led to the best possible decisions in the game.",
    "Our decisions helped move the story or mission toward success.",
    "I was satisfied with the results of our decisions.",
    "Overall, I believe we handled the game\u2019s decision-making challenges effectively."
]

# Coerce all scale columns to numeric (some have curly-quote in name causing object dtype)
scale_cols = GT_COLS + AI_COLS + SAT_COLS + [
    "How insecure, discouraged, irritated, stressed, or annoyed did you feel during the task?",
    "Our team is united in trying to reach its goals for performance.",
]
for c in scale_cols:
    if c in post.columns:
        post[c] = pd.to_numeric(post[c], errors="coerce")

post["groupthink_score"]   = post[GT_COLS].mean(axis=1)
post["ai_influence_score"] = post[AI_COLS].mean(axis=1)
post["satisfaction_score"] = post[SAT_COLS].mean(axis=1)
post["stress_score"]       = post["How insecure, discouraged, irritated, stressed, or annoyed did you feel during the task?"]
post["team_unity"]         = post["Our team is united in trying to reach its goals for performance."]
post["regret"]             = post["Do you regret any of the decisions made during the game?"].apply(
    lambda x: 1 if str(x).strip().lower()=="yes" else 0)

# 4. Build pre-survey features────────────────────────────────────────────
FAM_MAP  = {"Not familiar at all":1,"Slightly familiar":2,
            "Moderately familiar":3,"Very familiar":4,"Expert":5}
GAME_MAP = {"Never":0,"Rarely (a few times a year)":1,
            "Occasionally (monthly)":2,"Regularly (weekly or more)":3}

pre["ai_fam_num"]      = pre["Familiarity with AI (e.g., chatbots, voice assistants, decision-support tools):"].map(FAM_MAP)
pre["gaming_freq_num"] = pre["If yes, how often do you play?"].map(GAME_MAP)
pre["age_group"]       = pd.cut(pre["Age"], bins=[0,21,25,35,100],
                                labels=["18-21","22-25","26-35","36+"])

# 5. Merge
merged = (pre
    .merge(post[["Team ID","groupthink_score","ai_influence_score",
                 "satisfaction_score","stress_score","team_unity","regret"]],
           on="Team ID", how="inner")
    .merge(obs_g[["Team ID","ai_compliance_rate","dissent_rate",
                  "suppression_rate","ai_driven"]], on="Team ID", how="left")
    .merge(alive_g, on="Team ID", how="left"))

merged["compliance_group"] = pd.cut(merged["ai_compliance_rate"],
    bins=[-0.01,0.5,0.8,1.01], labels=["Low(<50%)","Mid(50-80%)","High(>80%)"])

# Ordered condition list for consistent display
COND_ORDER = ["Control",
              "Open Positive","Open Negative",
              "Closed Positive","Closed Negative",
              "Neutral Positive","Neutral Negative"]
FRAMING_ORDER = ["Control","Open","Closed","Neutral"]
VALENCE_ORDER = ["Control","Positive","Negative"]

print("="*70)
print(f"Merged dataset: {merged.shape[0]} participants, {merged.shape[1]} columns")
print("Condition distribution:")
print(merged["condition_norm"].value_counts().to_string())
print("="*70)

# 6. Statistical tests helper─────────────────────────────────────────────
def kruskal_report(label, groups_dict):
    groups = [v.dropna().values for v in groups_dict.values()]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return
    H, p = kruskal(*groups)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  Kruskal-Wallis  H={H:.2f}  p={p:.4f}  {sig}   [{label}]")
    for name, vals in groups_dict.items():
        v = vals.dropna()
        print(f"    {name:22s}  n={len(v):2d}  median={v.median():.2f}  mean={v.mean():.2f}")

def spearman_report(label, x, y):
    mask = x.notna() & y.notna()
    x2, y2 = x[mask], y[mask]
    if len(x2) < 5: return
    r, p = spearmanr(x2, y2)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  Spearman  r={r:.3f}  p={p:.4f}  {sig}   n={len(x2)}   [{label}]")

def mwu_report(label, g1_vals, g2_vals, g1_name="A", g2_name="B"):
    a, b = g1_vals.dropna(), g2_vals.dropna()
    if len(a)<2 or len(b)<2: return
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  Mann-Whitney U  U={U:.1f}  p={p:.4f}  {sig}")
    print(f"    {g1_name:22s}  n={len(a):2d}  median={a.median():.2f}  mean={a.mean():.2f}")
    print(f"    {g2_name:22s}  n={len(b):2d}  median={b.median():.2f}  mean={b.mean():.2f}")
    print(f"  [{label}]")

# 7. Run all statistical tests────────────────────────────────────────────
print("\n" + "="*70)
print("FINDING 1 — AGE GROUP vs GROUPTHINK, AI COMPLIANCE, SURVIVORS")
print("="*70)
for metric in ["groupthink_score","ai_compliance_rate","survivors"]:
    groups = {g: merged.loc[merged["age_group"]==g, metric]
              for g in ["18-21","22-25","26-35","36+"]}
    kruskal_report(f"Age group vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 2 — AI FAMILIARITY vs GROUPTHINK & COMPLIANCE")
print("="*70)
for metric in ["groupthink_score","ai_compliance_rate","regret"]:
    groups = {str(k): merged.loc[merged["ai_fam_num"]==k, metric]
              for k in sorted(merged["ai_fam_num"].dropna().unique())}
    kruskal_report(f"AI familiarity vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 3 — GAMING FREQUENCY vs GROUPTHINK, COMPLIANCE, SATISFACTION")
print("="*70)
for metric in ["groupthink_score","ai_compliance_rate","satisfaction_score"]:
    groups = {str(k): merged.loc[merged["gaming_freq_num"]==k, metric]
              for k in sorted(merged["gaming_freq_num"].dropna().unique())}
    kruskal_report(f"Gaming freq vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 4 — COMPLIANCE TIER vs SURVIVORS, SATISFACTION, STRESS")
print("="*70)
for metric in ["survivors","satisfaction_score","stress_score","regret"]:
    groups = {g: merged.loc[merged["compliance_group"]==g, metric]
              for g in ["Low(<50%)","Mid(50-80%)","High(>80%)"]}
    kruskal_report(f"Compliance tier vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 5 — GENDER vs GROUPTHINK, AI COMPLIANCE, REGRET")
print("="*70)
m_df = merged[merged["Gender"]=="Male"]
f_df = merged[merged["Gender"]=="Female"]
for metric in ["groupthink_score","ai_compliance_rate","ai_influence_score","regret","satisfaction_score"]:
    print(f"\n  {metric}:")
    mwu_report(f"Gender vs {metric}", m_df[metric], f_df[metric], "Male","Female")

print("\n" + "="*70)
print("FINDING 6 — SPEARMAN CORRELATIONS (cross-file)")
print("="*70)
pairs = [
    ("Age",               "groupthink_score",   "Age vs Groupthink"),
    ("Age",               "ai_compliance_rate", "Age vs AI Compliance"),
    ("Age",               "survivors",          "Age vs Survivors"),
    ("ai_fam_num",        "groupthink_score",   "AI Familiarity vs Groupthink"),
    ("ai_fam_num",        "ai_compliance_rate", "AI Familiarity vs Compliance"),
    ("ai_fam_num",        "regret",             "AI Familiarity vs Regret"),
    ("gaming_freq_num",   "groupthink_score",   "Gaming vs Groupthink"),
    ("gaming_freq_num",   "satisfaction_score", "Gaming vs Satisfaction"),
    ("ai_compliance_rate","survivors",          "Compliance vs Survivors"),
    ("ai_compliance_rate","satisfaction_score", "Compliance vs Satisfaction"),
    ("groupthink_score",  "ai_influence_score", "Groupthink vs AI Influence"),
    ("groupthink_score",  "satisfaction_score", "Groupthink vs Satisfaction"),
    ("stress_score",      "regret",             "Stress vs Regret"),
    ("satisfaction_score","regret",             "Satisfaction vs Regret"),
]
for x_col, y_col, label in pairs:
    spearman_report(label, merged[x_col], merged[y_col])

# NEW FINDINGS 7-9──────
print("\n" + "="*70)
print("FINDING 7 — CONDITION (full 7-level) vs KEY OUTCOMES")
print("="*70)
cond_present = [c for c in COND_ORDER if c in merged["condition_norm"].values]
for metric in ["groupthink_score","ai_compliance_rate","dissent_rate",
               "survivors","satisfaction_score","stress_score"]:
    groups = {c: merged.loc[merged["condition_norm"]==c, metric]
              for c in cond_present}
    kruskal_report(f"Condition vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 8 — FRAMING (Open/Closed/Neutral/Control) vs KEY OUTCOMES")
print("="*70)
framing_present = [f for f in FRAMING_ORDER if f in merged["framing"].values]
for metric in ["ai_compliance_rate","dissent_rate","suppression_rate",
               "groupthink_score","ai_influence_score"]:
    groups = {f: merged.loc[merged["framing"]==f, metric]
              for f in framing_present}
    kruskal_report(f"Framing vs {metric}", groups)

print("\n" + "="*70)
print("FINDING 9 — VALENCE (Positive/Negative/Control) vs KEY OUTCOMES")
print("="*70)
val_present = [v for v in VALENCE_ORDER if v in merged["valence"].values]
for metric in ["groupthink_score","satisfaction_score","regret",
               "stress_score","ai_compliance_rate","survivors"]:
    groups = {v: merged.loc[merged["valence"]==v, metric]
              for v in val_present}
    kruskal_report(f"Valence vs {metric}", groups)

# 8. Print raw group means────────────────────────────────────────────────
print("\n" + "="*70)
print("RAW GROUP MEANS (what the dashboard charts show)")
print("="*70)

print("\n-- Age group --")
print(merged.groupby("age_group", observed=True)[
    ["groupthink_score","ai_influence_score","ai_compliance_rate","survivors","regret"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- AI Familiarity --")
print(merged.groupby("ai_fam_num")[
    ["groupthink_score","ai_compliance_rate","satisfaction_score","regret","survivors"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Gaming frequency --")
print(merged.groupby("gaming_freq_num")[
    ["groupthink_score","ai_compliance_rate","satisfaction_score","dissent_rate"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Compliance tier --")
print(merged.groupby("compliance_group", observed=True)[
    ["survivors","satisfaction_score","stress_score","regret","Age","ai_fam_num"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Gender --")
print(merged.groupby("Gender")[
    ["groupthink_score","ai_compliance_rate","ai_influence_score","dissent_rate","satisfaction_score","regret"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Condition (full) --")
print(merged.groupby("condition_norm")[
    ["groupthink_score","ai_compliance_rate","dissent_rate","survivors","satisfaction_score","stress_score"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Framing --")
print(merged.groupby("framing")[
    ["ai_compliance_rate","dissent_rate","groupthink_score","ai_influence_score","suppression_rate"]
].agg(["mean","count"]).round(2).to_string())

print("\n-- Valence --")
print(merged.groupby("valence")[
    ["groupthink_score","satisfaction_score","regret","stress_score","ai_compliance_rate"]
].agg(["mean","count"]).round(2).to_string())

# 9. Plots
COLORS = {
    "blue":   "#3266ad",
    "coral":  "#D85A30",
    "teal":   "#1D9E75",
    "amber":  "#BA7517",
    "purple": "#534AB7",
    "pink":   "#D4537E",
    "gray":   "#888780",
    "green":  "#639922",
    "red":    "#A32D2D",
}

COND_COLORS = {
    "Control":          COLORS["gray"],
    "Open Positive":    COLORS["teal"],
    "Open Negative":    COLORS["coral"],
    "Closed Positive":  COLORS["blue"],
    "Closed Negative":  COLORS["red"],
    "Neutral Positive": COLORS["amber"],
    "Neutral Negative": COLORS["purple"],
}

def bar_group(ax, labels, datasets, title, ylabel="", ylim=None):
    x = np.arange(len(labels))
    w = 0.8 / len(datasets)
    for i, (name, vals, color) in enumerate(datasets):
        ax.bar(x + i*w - 0.4 + w/2, vals, width=w*0.9,
               label=name, color=color, zorder=3)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if ylim: ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)

fig = plt.figure(figsize=(18, 32))
fig.suptitle("Until Dawn — Cross-file Analysis v3 (with Condition)", fontsize=14, y=0.99)
gs  = gridspec.GridSpec(6, 2, figure=fig, hspace=0.6, wspace=0.35)

# Row 0: Age plots (unchanged)────────────────────────────────────────────
age_labels = ["18-21","22-25","26-35","36+"]
age_gt   = merged.groupby("age_group", observed=True)["groupthink_score"].mean().values
age_ai   = merged.groupby("age_group", observed=True)["ai_influence_score"].mean().values
age_comp = merged.groupby("age_group", observed=True)["ai_compliance_rate"].mean().values * 100
age_surv = merged.groupby("age_group", observed=True)["survivors"].mean().values

ax1 = fig.add_subplot(gs[0,0])
bar_group(ax1, age_labels,
          [("Groupthink /10", age_gt,   COLORS["blue"]),
           ("AI Influence /10", age_ai, COLORS["coral"])],
          "Age group: groupthink & AI influence", ylim=(0,10))

ax2 = fig.add_subplot(gs[0,1])
ax2b = ax2.twinx()
ax2.bar(np.arange(4)-0.2, age_comp, 0.35, color=COLORS["teal"],  label="AI compliance %", zorder=3)
ax2b.bar(np.arange(4)+0.2, age_surv, 0.35, color=COLORS["amber"], label="Avg survivors",   zorder=3, alpha=0.85)
ax2.set_title("Age group: AI compliance & survivors", fontsize=10, pad=6)
ax2.set_xticks(range(4)); ax2.set_xticklabels(age_labels, fontsize=8)
ax2.set_ylabel("Compliance %", fontsize=8); ax2b.set_ylabel("Survivors", fontsize=8)
ax2.set_ylim(0,110); ax2b.set_ylim(0,4)
ax2.grid(axis="y", alpha=0.3, zorder=0); ax2.spines[["top","right"]].set_visible(False)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=7)

# Row 1: Familiarity + Gaming (unchanged)─────────────────────────────────
fam_labels = ["Slightly(2)","Moderate(3)","Very(4)","Expert(5)"]
fam_vals   = merged.groupby("ai_fam_num")[["groupthink_score","ai_compliance_rate","regret"]].mean()
ax3 = fig.add_subplot(gs[1,0])
bar_group(ax3, fam_labels,
          [("Groupthink /10",  fam_vals["groupthink_score"].values,      COLORS["purple"]),
           ("Compliance ×10",  fam_vals["ai_compliance_rate"].values*10, COLORS["teal"]),
           ("Regret rate ×10", fam_vals["regret"].values*10,             COLORS["coral"])],
          "AI familiarity: groupthink, compliance & regret", ylim=(0,12))

game_labels = ["Rarely","Occasional","Regular"]
game_vals   = merged.groupby("gaming_freq_num")[["groupthink_score","ai_compliance_rate","satisfaction_score"]].mean()
ax4 = fig.add_subplot(gs[1,1])
bar_group(ax4, game_labels,
          [("Groupthink /10",    game_vals["groupthink_score"].values,      COLORS["purple"]),
           ("Satisfaction /10",  game_vals["satisfaction_score"].values,    COLORS["amber"]),
           ("AI compliance ×10", game_vals["ai_compliance_rate"].values*10, COLORS["teal"])],
          "Gaming freq: groupthink, satisfaction & compliance", ylim=(0,12))

# Row 2: Compliance tier + Gender (unchanged)─────────────────────────────
comp_labels = ["Low(<50%)","Mid(50-80%)","High(>80%)"]
comp_vals   = merged.groupby("compliance_group", observed=True)[
    ["survivors","satisfaction_score","stress_score","regret"]].mean()
ax5 = fig.add_subplot(gs[2,0])
bar_group(ax5, comp_labels,
          [("Survivors",        comp_vals["survivors"].values,          COLORS["teal"]),
           ("Satisfaction /10", comp_vals["satisfaction_score"].values, COLORS["blue"]),
           ("Stress /10",       comp_vals["stress_score"].values,       COLORS["coral"]),
           ("Regret ×10",       comp_vals["regret"].values*10,          COLORS["amber"])],
          "Compliance tier: survivors, satisfaction, stress, regret", ylim=(0,10))

gen_labels = ["Groupthink","AI compliance\n×10","AI influence","Dissent\n×10","Satisfaction","Regret\n×10"]
f_vals = [
    merged[merged["Gender"]=="Female"]["groupthink_score"].mean(),
    merged[merged["Gender"]=="Female"]["ai_compliance_rate"].mean()*10,
    merged[merged["Gender"]=="Female"]["ai_influence_score"].mean(),
    merged[merged["Gender"]=="Female"]["dissent_rate"].mean()*10,
    merged[merged["Gender"]=="Female"]["satisfaction_score"].mean(),
    merged[merged["Gender"]=="Female"]["regret"].mean()*10,
]
m_vals = [
    merged[merged["Gender"]=="Male"]["groupthink_score"].mean(),
    merged[merged["Gender"]=="Male"]["ai_compliance_rate"].mean()*10,
    merged[merged["Gender"]=="Male"]["ai_influence_score"].mean(),
    merged[merged["Gender"]=="Male"]["dissent_rate"].mean()*10,
    merged[merged["Gender"]=="Male"]["satisfaction_score"].mean(),
    merged[merged["Gender"]=="Male"]["regret"].mean()*10,
]
ax6 = fig.add_subplot(gs[2,1])
bar_group(ax6, gen_labels,
          [("Female", f_vals, COLORS["pink"]),
           ("Male",   m_vals, COLORS["blue"])],
          "Gender: behavioral & outcome metrics", ylim=(0,12))

# Row 3: NEW — Condition (full 7-level) grouped bar──────────────────────
ax_c1 = fig.add_subplot(gs[3,0])
cond_present = [c for c in COND_ORDER if c in merged["condition_norm"].values]
cond_gt   = [merged[merged["condition_norm"]==c]["groupthink_score"].mean()   for c in cond_present]
cond_ai   = [merged[merged["condition_norm"]==c]["ai_influence_score"].mean() for c in cond_present]
bar_group(ax_c1, [c.replace(" ","\n") for c in cond_present],
          [("Groupthink /10",   cond_gt, COLORS["blue"]),
           ("AI Influence /10", cond_ai, COLORS["coral"])],
          "Condition: groupthink & AI influence", ylim=(0,10))

ax_c2 = fig.add_subplot(gs[3,1])
cond_comp  = [merged[merged["condition_norm"]==c]["ai_compliance_rate"].mean()*100 for c in cond_present]
cond_diss  = [merged[merged["condition_norm"]==c]["dissent_rate"].mean()*100       for c in cond_present]
bar_group(ax_c2, [c.replace(" ","\n") for c in cond_present],
          [("AI compliance %", cond_comp, COLORS["teal"]),
           ("Dissent rate %",  cond_diss, COLORS["coral"])],
          "Condition: AI compliance & dissent rate", ylim=(0,110))

# Row 4: NEW — Framing bars + Valence bars───────────────────────────────
ax_f = fig.add_subplot(gs[4,0])
framing_present = [f for f in FRAMING_ORDER if f in merged["framing"].values]
fr_comp = [merged[merged["framing"]==f]["ai_compliance_rate"].mean()*100 for f in framing_present]
fr_diss = [merged[merged["framing"]==f]["dissent_rate"].mean()*100       for f in framing_present]
fr_gt   = [merged[merged["framing"]==f]["groupthink_score"].mean()       for f in framing_present]
bar_group(ax_f, framing_present,
          [("AI compliance %",  fr_comp, COLORS["teal"]),
           ("Dissent rate %",   fr_diss, COLORS["coral"]),
           ("Groupthink ×10",   [v*10 for v in fr_gt], COLORS["purple"])],
          "Framing type: compliance, dissent & groupthink",
          ylim=(0,120))

ax_v = fig.add_subplot(gs[4,1])
val_present = [v for v in VALENCE_ORDER if v in merged["valence"].values]
val_gt   = [merged[merged["valence"]==v]["groupthink_score"].mean()   for v in val_present]
val_sat  = [merged[merged["valence"]==v]["satisfaction_score"].mean() for v in val_present]
val_str  = [merged[merged["valence"]==v]["stress_score"].mean()       for v in val_present]
val_reg  = [merged[merged["valence"]==v]["regret"].mean()*10          for v in val_present]
bar_group(ax_v, val_present,
          [("Groupthink /10",    val_gt,  COLORS["blue"]),
           ("Satisfaction /10",  val_sat, COLORS["amber"]),
           ("Stress /10",        val_str, COLORS["coral"]),
           ("Regret ×10",        val_reg, COLORS["purple"])],
          "Valence: groupthink, satisfaction, stress & regret",
          ylim=(0,12))

# Row 5: Correlation heatmap (full width, now includes condition_encoded)
corr_cols = ["Age","ai_fam_num","gaming_freq_num","groupthink_score",
             "ai_influence_score","ai_compliance_rate","dissent_rate",
             "satisfaction_score","stress_score","survivors","regret"]
corr_labels = ["Age","AI fam","Gaming","Groupthink","AI infl",
               "Compliance","Dissent","Satisf","Stress","Survive","Regret"]
corr_matrix = merged[corr_cols].corr(method="spearman")
ax7 = fig.add_subplot(gs[5,:])
im = ax7.imshow(corr_matrix.values, cmap="Oranges", vmin=-1, vmax=1, aspect="auto")
ax7.set_xticks(range(len(corr_labels))); ax7.set_xticklabels(corr_labels, fontsize=8, rotation=30)
ax7.set_yticks(range(len(corr_labels))); ax7.set_yticklabels(corr_labels, fontsize=8)
ax7.set_title("Spearman correlation heatmap (all cross-file variables)", fontsize=10, pad=6)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        v = corr_matrix.values[i,j]
        ax7.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=7, color="black" if abs(v)<0.6 else "white")
plt.colorbar(im, ax=ax7, fraction=0.02)

combined_path = os.path.join(output_dir, 'until_dawn_analysis_v3.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight')

print("\nDone.")
