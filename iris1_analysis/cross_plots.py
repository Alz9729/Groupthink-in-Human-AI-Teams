
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, spearmanr
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = "cross_plots"
os.makedirs(OUT_DIR, exist_ok=True)

OBS_FILE  = "Observation Sheet_April 6, 2026_12.19.xlsx"
PRE_FILE  = "Until Dawn Pre-survey_April 6, 2026_12.17.xlsx"
POST_FILE = "Until Dawn Post Survey_April 6, 2026_12.18.xlsx"

#Palette────────────
C = dict(blue="#3266ad", coral="#D85A30", teal="#1D9E75", amber="#BA7517",
         purple="#534AB7", pink="#D4537E", gray="#888780", green="#639922",
         red="#A32D2D")

COND_COLORS = {
    "Control":          C["gray"],
    "Open Positive":    C["teal"],
    "Open Negative":    C["coral"],
    "Closed Positive":  C["blue"],
    "Closed Negative":  C["red"],
    "Neutral Positive": C["amber"],
    "Neutral Negative": C["purple"],
}

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,
                     "grid.alpha":0.25,"figure.dpi":140})

def sig_label(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"


# 1. LOAD & BUILD MERGED DATASET

pre  = pd.read_excel(PRE_FILE,  header=1)
post = pd.read_excel(POST_FILE, header=1)
obs  = pd.read_excel(OBS_FILE,  header=1)
for df in [pre, post, obs]:
    df["Team ID"] = pd.to_numeric(df["Team ID"], errors="coerce")

#Normalise condition
def norm_cond(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s == "control": return "Control"
    parts = s.split()
    fm = {"open":"Open","closed":"Closed","neutral":"Neutral"}
    vm = {"positive":"Positive","negative":"Negative"}
    f   = fm.get(parts[0], parts[0].title()) if len(parts)>0 else ""
    val = vm.get(parts[1], parts[1].title()) if len(parts)>1 else ""
    return f"{f} {val}".strip()

pre["condition_norm"] = pre["condition"].apply(norm_cond)
pre["framing"] = pre["condition_norm"].apply(
    lambda c: "Control" if pd.isna(c) or c=="Control" else c.split()[0])
pre["valence"] = pre["condition_norm"].apply(
    lambda c: "Control" if pd.isna(c) or c=="Control" else (c.split()[1] if len(c.split())>1 else "Unknown"))

COND_ORDER    = ["Control","Open Positive","Open Negative",
                 "Closed Positive","Closed Negative",
                 "Neutral Positive","Neutral Negative"]
FRAMING_ORDER = ["Control","Open","Closed","Neutral"]
VALENCE_ORDER = ["Control","Positive","Negative"]

#Observation metrics
agree_cols      = [c for c in obs.columns if "Agreement with AI"     in c]
disagree_cols   = [c for c in obs.columns if "Disagreement with AI"  in c]
source_cols     = [c for c in obs.columns if "Final decision source" in c]
dissent_cols    = [c for c in obs.columns if "Dissent Expressed"     in c]
suppressed_cols = [c for c in obs.columns if "ignored/suppressed"    in c]

def yn(v):
    if isinstance(v,str):
        s=v.strip().lower()
        if s in ["yes","yea","y"]: return 1
        if s in ["no","n"]:        return 0
    return np.nan

for col in agree_cols+disagree_cols+dissent_cols+suppressed_cols:
    obs[col]=obs[col].apply(yn)

obs["total_agree"]      = obs[agree_cols].sum(axis=1)
obs["total_dissent"]    = obs[dissent_cols].sum(axis=1)
obs["total_suppressed"] = obs[suppressed_cols].sum(axis=1)
obs["n_decisions"]      = obs[agree_cols].notna().sum(axis=1)
obs["ai_driven"]        = obs.apply(
    lambda r: sum(1 for c in source_cols
                  if pd.notna(r[c]) and str(r[c]).strip().upper() in ["AI","A.I."]), axis=1)

obs_g = obs.groupby("Team ID")[
    ["total_agree","total_dissent","total_suppressed","n_decisions","ai_driven"]
].mean().reset_index()
obs_g["ai_compliance_rate"] = obs_g["total_agree"]      / obs_g["n_decisions"]
obs_g["dissent_rate"]       = obs_g["total_dissent"]    / obs_g["n_decisions"]
obs_g["suppression_rate"]   = obs_g["total_suppressed"] / obs_g["n_decisions"]

def parse_alive(text):
    if pd.isna(text): return np.nan
    t=str(text).lower()
    for n in ["4","3","2","1","0"]:
        if n+" person" in t or n+" people" in t: return int(n)
    if "all" in t and "saved" in t: return 4
    return np.nan

obs["alive_parsed"] = obs["No of people alive"].apply(parse_alive)
alive_g = obs.groupby("Team ID")["alive_parsed"].mean().reset_index()
alive_g.columns = ["Team ID","survivors"]

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
    "How mentally demanding was the task?",
    "How hurried or rushed was the pace of the task?",
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
post["mental_demand"]      = post["How mentally demanding was the task?"]
post["time_pressure"]      = post["How hurried or rushed was the pace of the task?"]

FAM_MAP  = {"Not familiar at all":1,"Slightly familiar":2,
            "Moderately familiar":3,"Very familiar":4,"Expert":5}
GAME_MAP = {"Never":0,"Rarely (a few times a year)":1,
            "Occasionally (monthly)":2,"Regularly (weekly or more)":3}

pre["ai_fam_num"]      = pre["Familiarity with AI (e.g., chatbots, voice assistants, decision-support tools):"].map(FAM_MAP)
pre["gaming_freq_num"] = pre["If yes, how often do you play?"].map(GAME_MAP)
pre["age_group"]       = pd.cut(pre["Age"],bins=[0,21,25,35,100],
                                labels=["18-21","22-25","26-35","36+"])

merged = (pre
    .merge(post[["Team ID","groupthink_score","ai_influence_score","satisfaction_score",
                 "stress_score","team_unity","regret","mental_demand","time_pressure"]],
           on="Team ID",how="inner")
    .merge(obs_g[["Team ID","ai_compliance_rate","dissent_rate",
                  "suppression_rate","ai_driven"]], on="Team ID",how="left")
    .merge(alive_g, on="Team ID",how="left"))

merged["compliance_group"] = pd.cut(merged["ai_compliance_rate"],
    bins=[-0.01,0.5,0.8,1.01], labels=["Low(<50%)","Mid(50-80%)","High(>80%)"])

print(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} cols\n")


# STATISTICAL TESTS

def kw(label, groups_dict):
    gs2 = [v.dropna().values for v in groups_dict.values()]
    if len(gs2)<2 or any(len(g)<2 for g in gs2): return None, None
    H,p = kruskal(*gs2)
    sig = sig_label(p)
    print(f"  Kruskal-Wallis H={H:.2f} p={p:.4f} {sig}  [{label}]")
    for name,vals in groups_dict.items():
        v=vals.dropna()
        print(f"    {name:22s} n={len(v):2d}  median={v.median():.2f}  mean={v.mean():.2f}")
    return H,p

def sp(label, x, y):
    mask=x.notna()&y.notna(); x2,y2=x[mask],y[mask]
    if len(x2)<5: return None,None
    r,p=spearmanr(x2,y2)
    print(f"  Spearman r={r:.3f} p={p:.4f} {sig_label(p)}  n={len(x2)}  [{label}]")
    return r,p

def mwu(label, a_s, b_s, an="A", bn="B"):
    a,b=a_s.dropna(),b_s.dropna()
    if len(a)<2 or len(b)<2: return None,None
    U,p=mannwhitneyu(a,b,alternative="two-sided")
    print(f"  MannWhitneyU U={U:.1f} p={p:.4f} {sig_label(p)}")
    print(f"    {an:22s} n={len(a):2d} median={a.median():.2f} mean={a.mean():.2f}")
    print(f"    {bn:22s} n={len(b):2d} median={b.median():.2f} mean={b.mean():.2f}")
    print(f"  [{label}]")
    return U,p

age_groups = ["18-21","22-25","26-35","36+"]

print("="*60); print("F1 — AGE vs GROUPTHINK / SURVIVORS"); print("="*60)
_,p_age_gt   = kw("age vs groupthink", {g:merged.loc[merged["age_group"]==g,"groupthink_score"] for g in age_groups})
_,p_age_surv = kw("age vs survivors",  {g:merged.loc[merged["age_group"]==g,"survivors"]         for g in age_groups})

print("\n"+"="*60); print("F2 — AI FAMILIARITY vs GROUPTHINK / REGRET"); print("="*60)
fam_levels = sorted(merged["ai_fam_num"].dropna().unique())
_,p_fam_gt  = kw("fam vs groupthink", {str(k):merged.loc[merged["ai_fam_num"]==k,"groupthink_score"] for k in fam_levels})
_,p_fam_reg = kw("fam vs regret",     {str(k):merged.loc[merged["ai_fam_num"]==k,"regret"]           for k in fam_levels})

print("\n"+"="*60); print("F3 — GAMING FREQUENCY vs GROUPTHINK / SATISFACTION"); print("="*60)
game_levels = sorted(merged["gaming_freq_num"].dropna().unique())
_,p_game_gt  = kw("gaming vs groupthink",    {str(k):merged.loc[merged["gaming_freq_num"]==k,"groupthink_score"]   for k in game_levels})
_,p_game_sat = kw("gaming vs satisfaction",  {str(k):merged.loc[merged["gaming_freq_num"]==k,"satisfaction_score"] for k in game_levels})

print("\n"+"="*60); print("F4 — AI COMPLIANCE vs SURVIVORS"); print("="*60)
r_comp_surv, p_comp_surv = sp("compliance vs survivors", merged["ai_compliance_rate"], merged["survivors"])

print("\n"+"="*60); print("F5 — GENDER vs REGRET / COMPLIANCE"); print("="*60)
m = merged[merged["Gender"]=="Male"]
f = merged[merged["Gender"]=="Female"]
_,p_gen_reg  = mwu("gender vs regret",     m["regret"],             f["regret"],             "Male","Female")
_,p_gen_comp = mwu("gender vs compliance", m["ai_compliance_rate"], f["ai_compliance_rate"], "Male","Female")

print("\n"+"="*60); print("F7 — CONDITION (full) vs KEY OUTCOMES"); print("="*60)
cond_present = [c for c in COND_ORDER if c in merged["condition_norm"].values]
for metric in ["groupthink_score","ai_compliance_rate","dissent_rate","survivors","satisfaction_score"]:
    kw(f"condition vs {metric}", {c:merged.loc[merged["condition_norm"]==c, metric] for c in cond_present})

print("\n"+"="*60); print("F8 — FRAMING vs COMPLIANCE / DISSENT / GROUPTHINK"); print("="*60)
framing_present = [f for f in FRAMING_ORDER if f in merged["framing"].values]
for metric in ["ai_compliance_rate","dissent_rate","groupthink_score","ai_influence_score"]:
    kw(f"framing vs {metric}", {f:merged.loc[merged["framing"]==f, metric] for f in framing_present})

print("\n"+"="*60); print("F9 — VALENCE vs GROUPTHINK / SATISFACTION / REGRET"); print("="*60)
val_present = [v for v in VALENCE_ORDER if v in merged["valence"].values]
for metric in ["groupthink_score","satisfaction_score","regret","stress_score"]:
    kw(f"valence vs {metric}", {v:merged.loc[merged["valence"]==v, metric] for v in val_present})

print("\n"+"="*60); print("F6 — SPEARMAN CORRELATIONS"); print("="*60)
corr_cols = ["Age","ai_fam_num","gaming_freq_num","groupthink_score",
             "ai_influence_score","ai_compliance_rate","dissent_rate",
             "satisfaction_score","stress_score","survivors","regret"]
pairs=[
    ("Age","groupthink_score","Age vs Groupthink"),
    ("Age","survivors","Age vs Survivors"),
    ("ai_fam_num","regret","AI Fam vs Regret"),
    ("gaming_freq_num","groupthink_score","Gaming vs Groupthink"),
    ("gaming_freq_num","satisfaction_score","Gaming vs Satisfaction"),
    ("ai_compliance_rate","survivors","Compliance vs Survivors"),
    ("groupthink_score","ai_influence_score","Groupthink vs AI Influence"),
    ("groupthink_score","satisfaction_score","Groupthink vs Satisfaction"),
    ("satisfaction_score","regret","Satisfaction vs Regret"),
    ("stress_score","groupthink_score","Stress vs Groupthink"),
    ("mental_demand","ai_compliance_rate","Mental demand vs Compliance"),
    ("time_pressure","dissent_rate","Time pressure vs Dissent"),
    ("team_unity","satisfaction_score","Team unity vs Satisfaction"),
    ("suppression_rate","survivors","Suppression vs Survivors"),
]
for x,y,lbl in pairs:
    sp(lbl, merged[x], merged[y])


# PLOTS 1-12 


#PLOT 1: Age vs Groupthink & AI Influence — BOXPLOT
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 1 — Age group vs Groupthink & AI Influence\n(Kruskal-Wallis, boxplot)", fontsize=11,y=1.01)
for ax,col,title,color in [
    (axes[0],"groupthink_score","Groupthink score",C["blue"]),
    (axes[1],"ai_influence_score","AI influence score",C["coral"])]:
    data=[merged.loc[merged["age_group"]==g,col].dropna().values for g in age_groups]
    bp=ax.boxplot(data, patch_artist=True, notch=False,
                  medianprops=dict(color="white",linewidth=2),
                  whiskerprops=dict(color=color), capprops=dict(color=color),
                  flierprops=dict(marker="o",markerfacecolor=color,markersize=4,alpha=0.5))
    for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(age_groups,fontsize=9)
    ax.set_xlabel("Age group"); ax.set_ylabel("Score (0–10)"); ax.set_title(title); ax.set_ylim(-0.5,12)
    H_v,p_v = kruskal(*[d for d in data if len(d)>=1])
    ax.text(0.97,0.97,f"KW: p={p_v:.3f} {sig_label(p_v)}",transform=ax.transAxes,
            ha="right",va="top",fontsize=8,color=C["teal"] if p_v<0.05 else C["gray"])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot1_age_groupthink_boxplot.png",bbox_inches="tight"); plt.close()
print("\nSaved plot1")

#PLOT 2: Age vs Survivors — VIOLIN
fig,ax=plt.subplots(figsize=(8,5))
surv_data=[merged.loc[merged["age_group"]==g,"survivors"].dropna().values for g in age_groups]
parts=ax.violinplot(surv_data, positions=range(len(age_groups)), showmedians=True, showmeans=False)
for i,pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(C.values())[i]); pc.set_alpha(0.7)
parts["cmedians"].set_colors([C["gray"]]*4); parts["cmedians"].set_linewidth(2)
for i,g in enumerate(age_groups):
    y=merged.loc[merged["age_group"]==g,"survivors"].dropna()
    ax.scatter(np.random.normal(i,0.05,len(y)),y,color=list(C.values())[i],s=30,zorder=5,alpha=0.8)
_,p_v=kruskal(*[d for d in surv_data if len(d)>=1])
ax.set_xticks(range(4)); ax.set_xticklabels(age_groups)
ax.set_ylabel("Survivors (out of 4)"); ax.set_xlabel("Age group")
ax.set_title(f"Plot 2 — Age group vs Survivors (KW p={p_v:.3f} {sig_label(p_v)})\nViolin = distribution, dots = individual teams")
ax.set_ylim(-0.5,5)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot2_age_survivors_violin.png",bbox_inches="tight"); plt.close()
print("Saved plot2")

#PLOT 3: AI Familiarity vs Groupthink & Regret — BOXPLOT + strip
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 3 — AI Familiarity vs Groupthink & Regret rate\n(Kruskal-Wallis; strip shows all data points)", fontsize=11,y=1.01)
fam_labels_map={2:"Slightly(2)",3:"Moderate(3)",4:"Very(4)",5:"Expert(5)"}
fam_x=[fam_labels_map[k] for k in fam_levels]
for ax,(col,title,color) in zip(axes,[
        ("groupthink_score","Groupthink score /10",C["purple"]),
        ("regret","Regret (0=No, 1=Yes)",C["coral"])]):
    data=[merged.loc[merged["ai_fam_num"]==k,col].dropna().values for k in fam_levels]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="white",linewidth=2),
                  whiskerprops=dict(color=color),capprops=dict(color=color),flierprops=dict(marker="",alpha=0))
    for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.6)
    for i,vals in enumerate(data):
        ax.scatter(np.random.normal(i+1,0.07,len(vals)),vals,color=color,s=22,zorder=5,alpha=0.85)
    ax.set_xticklabels(fam_x,fontsize=8); ax.set_xlabel("AI familiarity (pre-survey)"); ax.set_ylabel(title)
    H_v,p_v=kruskal(*[d for d in data if len(d)>=1])
    ax.set_title(f"{title}  (KW p={p_v:.3f} {sig_label(p_v)})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot3_aifamiliarity_groupthink_regret.png",bbox_inches="tight"); plt.close()
print("Saved plot3")

#PLOT 4: Gaming frequency vs Groupthink & Satisfaction — BOXPLOT
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 4 — Gaming frequency vs Groupthink & Satisfaction\n(Kruskal-Wallis)", fontsize=11,y=1.01)
game_labels_map={1:"Rarely",2:"Occasional",3:"Regular"}
game_x=[game_labels_map[k] for k in game_levels]
for ax,(col,title,color) in zip(axes,[
        ("groupthink_score","Groupthink score /10",C["purple"]),
        ("satisfaction_score","Satisfaction /10",C["amber"])]):
    data=[merged.loc[merged["gaming_freq_num"]==k,col].dropna().values for k in game_levels]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="white",linewidth=2),
                  whiskerprops=dict(color=color),capprops=dict(color=color),flierprops=dict(marker="",alpha=0))
    for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.65)
    for i,vals in enumerate(data):
        ax.scatter(np.random.normal(i+1,0.07,len(vals)),vals,color=color,s=22,zorder=5,alpha=0.85)
    ax.set_xticklabels(game_x,fontsize=9); ax.set_xlabel("Gaming frequency"); ax.set_ylabel(title)
    H_v,p_v=kruskal(*[d for d in data if len(d)>=1])
    ax.set_title(f"{title}  (KW p={p_v:.3f} {sig_label(p_v)})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot4_gaming_groupthink_satisfaction.png",bbox_inches="tight"); plt.close()
print("Saved plot4")

#PLOT 5: AI Compliance vs Survivors — SCATTER + regression
fig,ax=plt.subplots(figsize=(7,5))
sub=merged[["ai_compliance_rate","survivors","age_group"]].dropna()
colors_age={"18-21":C["blue"],"22-25":C["teal"],"26-35":C["amber"],"36+":C["coral"]}
for g,grp in sub.groupby("age_group",observed=True):
    ax.scatter(grp["ai_compliance_rate"]*100, grp["survivors"],
               color=colors_age[str(g)], s=55, label=str(g), zorder=4, alpha=0.85)
x2=sub["ai_compliance_rate"].values; y2=sub["survivors"].values
m_,b_=np.polyfit(x2,y2,1)
xr=np.linspace(x2.min(),x2.max(),100)
ax.plot(xr*100, m_*xr+b_, color=C["gray"], lw=1.8, ls="--", label="linear fit")
r_v,p_v=spearmanr(x2,y2)
ax.set_xlabel("AI compliance rate (%)"); ax.set_ylabel("Survivors (out of 4)")
ax.set_title(f"Plot 5 — AI compliance vs Survivors\nSpearman r={r_v:.3f}, p={p_v:.3f} {sig_label(p_v)}  (colored by age group)")
ax.legend(fontsize=8,title="Age group",title_fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot5_compliance_survivors_scatter.png",bbox_inches="tight"); plt.close()
print("Saved plot5")

#PLOT 6: Gender — HORIZONTAL BAR
metrics_gen=[
    ("groupthink_score",   "Groupthink /10"),
    ("ai_compliance_rate", "AI compliance (×10)"),
    ("ai_influence_score", "AI influence /10"),
    ("dissent_rate",       "Dissent rate (×10)"),
    ("satisfaction_score", "Satisfaction /10"),
    ("regret",             "Regret rate (×10)"),
]
fv=[merged[merged["Gender"]=="Female"][m].mean()*(10 if "rate" in m or m=="regret" else 1) for m,_ in metrics_gen]
mv=[merged[merged["Gender"]=="Male"  ][m].mean()*(10 if "rate" in m or m=="regret" else 1) for m,_ in metrics_gen]
labels_gen=[l for _,l in metrics_gen]
fig,ax=plt.subplots(figsize=(9,6))
y=np.arange(len(labels_gen)); h=0.35
ax.barh(y+h/2, mv, h, color=C["blue"],  label="Male",   alpha=0.85)
ax.barh(y-h/2, fv, h, color=C["pink"],  label="Female", alpha=0.85)
for i,(mv2,fv2,lbl) in enumerate(zip(mv,fv,labels_gen)):
    mc=metrics_gen[i][0]
    _,pv=mannwhitneyu(merged[merged["Gender"]=="Male"][mc].dropna(),
                       merged[merged["Gender"]=="Female"][mc].dropna(),alternative="two-sided")
    xmax=max(mv2,fv2)
    ax.text(xmax+0.1,i,f"p={pv:.3f} {sig_label(pv)}",va="center",fontsize=7.5,
            color=C["teal"] if pv<0.05 else C["gray"])
ax.set_yticks(y); ax.set_yticklabels(labels_gen,fontsize=9)
ax.set_xlabel("Score / scaled value"); ax.set_xlim(0,13)
ax.set_title("Plot 6 — Gender comparison across all key metrics\np-values from Mann-Whitney U test (right margin)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot6_gender_horizontal_bar.png",bbox_inches="tight"); plt.close()
print("Saved plot6")

#PLOT 7: Spearman correlation heatmap
fig,ax=plt.subplots(figsize=(9,8))
corr_matrix=merged[corr_cols].corr(method="spearman")
corr_labels=["Age","AI fam","Gaming","Groupthink","AI infl","Compliance","Dissent","Satisf","Stress","Survive","Regret"]
mask=np.zeros_like(corr_matrix,dtype=bool)
mask[np.triu_indices_from(mask,k=1)]=True
sns.heatmap(corr_matrix, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1,
            xticklabels=corr_labels, yticklabels=corr_labels,
            annot_kws={"size":8}, linewidths=0.4, linecolor="#e0e0e0",
            cbar_kws={"label":"Spearman r","shrink":0.7})
ax.set_title("Plot 7 — Spearman correlation heatmap (lower triangle)\nBlue = positive, Red = negative", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot7_correlation_heatmap.png",bbox_inches="tight"); plt.close()
print("Saved plot7")

#PLOT 8: Stress & Mental demand vs AI Compliance — scatter
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 8 — Task load (post-survey) vs AI compliance rate\nDoes feeling rushed/stressed push people toward the AI?",fontsize=11,y=1.01)
for ax,(xcol,xlabel) in zip(axes,[("stress_score","Stress / annoyance (0–10)"),("mental_demand","Mental demand (0–10)")]):
    sub2=merged[[xcol,"ai_compliance_rate","Gender"]].dropna()
    for gender,color in [("Male",C["blue"]),("Female",C["pink"])]:
        g=sub2[sub2["Gender"]==gender]
        ax.scatter(g[xcol], g["ai_compliance_rate"]*100, color=color,s=45,alpha=0.8,label=gender)
    x3=sub2[xcol].values; y3=sub2["ai_compliance_rate"].values
    m3,b3=np.polyfit(x3,y3,1); xr3=np.linspace(x3.min(),x3.max(),80)
    ax.plot(xr3,(m3*xr3+b3)*100,color=C["gray"],lw=1.5,ls="--")
    r3,p3=spearmanr(x3,y3)
    ax.set_xlabel(xlabel); ax.set_ylabel("AI compliance (%)")
    ax.set_title(f"Spearman r={r3:.3f}, p={p3:.3f} {sig_label(p3)}")
    ax.legend(fontsize=8); ax.set_ylim(-5,115)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot8_taskload_compliance_scatter.png",bbox_inches="tight"); plt.close()
print("Saved plot8")

#PLOT 9: Time pressure vs Dissent & Suppression — scatter
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 9 — Time pressure vs Dissent & Suppression rate\nWere people less likely to voice disagreement when rushed?",fontsize=11,y=1.01)
for ax,(ycol,ylabel,color) in zip(axes,[
        ("dissent_rate","Dissent rate (proportion of decisions)",C["coral"]),
        ("suppression_rate","Suppression rate (proportion of decisions)",C["purple"])]):
    sub3=merged[["time_pressure",ycol]].dropna()
    ax.scatter(sub3["time_pressure"],sub3[ycol]*100,color=color,s=45,alpha=0.8,edgecolors="white",lw=0.5)
    x4=sub3["time_pressure"].values; y4=sub3[ycol].values
    m4,b4=np.polyfit(x4,y4,1); xr4=np.linspace(x4.min(),x4.max(),80)
    ax.plot(xr4,(m4*xr4+b4)*100,color=C["gray"],lw=1.5,ls="--")
    r4,p4=spearmanr(x4,y4)
    ax.set_xlabel("Time pressure (0–10)"); ax.set_ylabel(ylabel+" (%)")
    ax.set_title(f"Spearman r={r4:.3f}, p={p4:.3f} {sig_label(p4)}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot9_timepressure_dissent_suppression.png",bbox_inches="tight"); plt.close()
print("Saved plot9")

#PLOT 10: Team Unity vs Satisfaction — bubble
fig,ax=plt.subplots(figsize=(7,5))
sub4=merged[["team_unity","satisfaction_score","groupthink_score","age_group"]].dropna()
sc=ax.scatter(sub4["team_unity"], sub4["satisfaction_score"],
              c=sub4["groupthink_score"], cmap="RdYlGn_r",
              s=sub4["groupthink_score"]*12+15, alpha=0.8, edgecolors="white", lw=0.5, vmin=0, vmax=10)
cbar=plt.colorbar(sc,ax=ax); cbar.set_label("Groupthink score",fontsize=9)
x5=sub4["team_unity"].values; y5=sub4["satisfaction_score"].values
m5,b5=np.polyfit(x5,y5,1); xr5=np.linspace(x5.min(),x5.max(),80)
ax.plot(xr5,m5*xr5+b5,color=C["gray"],lw=1.5,ls="--")
r5,p5=spearmanr(x5,y5)
ax.set_xlabel("Team unity (0–10)"); ax.set_ylabel("Satisfaction (0–10)")
ax.set_title(f"Plot 10 — Team unity vs Satisfaction\nBubble size & color = Groupthink  |  Spearman r={r5:.3f}, p={p5:.3f} {sig_label(p5)}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot10_teamunity_satisfaction_bubble.png",bbox_inches="tight"); plt.close()
print("Saved plot10")

#PLOT 11: Race breakdown
race_order=["White","Black or African American","Asian","Middle Eastern"]
race_present=[r for r in race_order if r in merged["Race"].values]
rgt=[merged[merged["Race"]==r]["groupthink_score"].mean() for r in race_present]
rcomp=[merged[merged["Race"]==r]["ai_compliance_rate"].mean()*100 for r in race_present]
rn=[len(merged[merged["Race"]==r]) for r in race_present]
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Plot 11 — Race/ethnicity vs Groupthink & AI compliance\n(small groups — interpret cautiously)",fontsize=11,y=1.01)
for ax,(vals,title,color) in zip(axes,[
        (rgt,"Groupthink score /10",C["purple"]),
        (rcomp,"AI compliance (%)",C["teal"])]):
    bars=ax.bar(range(len(race_present)),vals,color=color,alpha=0.8,edgecolor="white",lw=0.8)
    for i,(bar,n) in enumerate(zip(bars,rn)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,f"n={n}",
                ha="center",va="bottom",fontsize=8,color=C["gray"])
    ax.set_xticks(range(len(race_present)))
    ax.set_xticklabels([r.replace(" or ","\nor ") for r in race_present],fontsize=8)
    ax.set_ylabel(title); ax.set_title(title)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot11_race_groupthink_compliance.png",bbox_inches="tight"); plt.close()
print("Saved plot11")

#PLOT 12: Satisfaction vs Regret — boxplot split + strip
fig,ax=plt.subplots(figsize=(7,5))
reg_no  = merged[merged["regret"]==0]["satisfaction_score"].dropna()
reg_yes = merged[merged["regret"]==1]["satisfaction_score"].dropna()
bp=ax.boxplot([reg_no,reg_yes],patch_artist=True,
              medianprops=dict(color="white",linewidth=2.5),
              whiskerprops=dict(color=C["gray"]),capprops=dict(color=C["gray"]),
              flierprops=dict(marker="",alpha=0))
bp["boxes"][0].set_facecolor(C["teal"]); bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(C["coral"]); bp["boxes"][1].set_alpha(0.7)
for i,(vals,color) in enumerate([(reg_no,C["teal"]),(reg_yes,C["coral"])]):
    ax.scatter(np.random.normal(i+1,0.07,len(vals)),vals,color=color,s=30,zorder=5,alpha=0.85)
_,pv=mannwhitneyu(reg_no,reg_yes,alternative="two-sided")
ax.set_xticks([1,2]); ax.set_xticklabels(["No Regret","Regret"],fontsize=10)
ax.set_ylabel("Satisfaction score (0–10)")
ax.set_title(f"Plot 12 — Satisfaction vs Regret\nMWU p={pv:.3f} {sig_label(pv)}")
patch1=mpatches.Patch(color=C["teal"],alpha=0.7,label=f"No regret (n={len(reg_no)})")
patch2=mpatches.Patch(color=C["coral"],alpha=0.7,label=f"Regret (n={len(reg_yes)})")
ax.legend(handles=[patch1,patch2],fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot12_satisfaction_regret_boxplot.png",bbox_inches="tight"); plt.close()
print("Saved plot12")


# NEW PLOTS 13-16 — Condition analysis


#PLOT 13: Full condition comparison — heatmap of means
metrics_cond = [
    ("groupthink_score",   "Groupthink /10"),
    ("ai_influence_score", "AI influence /10"),
    ("ai_compliance_rate", "AI compliance"),
    ("dissent_rate",       "Dissent rate"),
    ("satisfaction_score", "Satisfaction /10"),
    ("stress_score",       "Stress /10"),
    ("survivors",          "Survivors"),
]
cond_present = [c for c in COND_ORDER if c in merged["condition_norm"].values]
heat_data = pd.DataFrame(
    {lbl: [merged[merged["condition_norm"]==c][col].mean() for c in cond_present]
     for col,lbl in metrics_cond},
    index=cond_present
)
fig,ax=plt.subplots(figsize=(11,6))
sns.heatmap(heat_data, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, linecolor="#dddddd", annot_kws={"size":9},
            cbar_kws={"label":"Mean value","shrink":0.7})
ax.set_title("Plot 13 (NEW) — Condition × Metric mean heatmap\n"
             "Rows = conditions  |  Columns = outcome metrics", fontsize=11)
ax.set_xlabel("Metric"); ax.set_ylabel("Condition")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot13_condition_heatmap.png",bbox_inches="tight"); plt.close()
print("Saved plot13")

#PLOT 14: Framing (Open/Closed/Neutral/Control) vs compliance, dissent, groupthink — BOXPLOT
framing_present = [f for f in FRAMING_ORDER if f in merged["framing"].values]
FRAMING_COLORS  = {"Control":C["gray"],"Open":C["teal"],"Closed":C["blue"],"Neutral":C["amber"]}

fig,axes=plt.subplots(1,3,figsize=(14,5))
fig.suptitle("Plot 14 (NEW) — Framing type vs compliance, dissent & groupthink\n"
             "(Kruskal-Wallis; Open=free text, Closed=constrained, Neutral=balanced)",
             fontsize=11,y=1.01)

for ax,(col,ylabel) in zip(axes,[
        ("ai_compliance_rate","AI compliance rate"),
        ("dissent_rate","Dissent rate"),
        ("groupthink_score","Groupthink score /10")]):
    data=[merged.loc[merged["framing"]==f,col].dropna().values for f in framing_present]
    bp=ax.boxplot(data,patch_artist=True,
                  medianprops=dict(color="white",linewidth=2),
                  whiskerprops=dict(color="#555"),capprops=dict(color="#555"),
                  flierprops=dict(marker="",alpha=0))
    for patch,f in zip(bp["boxes"],framing_present):
        patch.set_facecolor(FRAMING_COLORS[f]); patch.set_alpha(0.75)
    for i,(vals,f) in enumerate(zip(data,framing_present)):
        ax.scatter(np.random.normal(i+1,0.07,len(vals)),vals,
                   color=FRAMING_COLORS[f],s=28,zorder=5,alpha=0.85)
    ax.set_xticklabels(framing_present,fontsize=9); ax.set_xlabel("Framing"); ax.set_ylabel(ylabel)
    gs2=[d for d in data if len(d)>=2]
    if len(gs2)>=2:
        H_v,p_v=kruskal(*gs2)
        ax.text(0.97,0.97,f"KW p={p_v:.3f} {sig_label(p_v)}",
                transform=ax.transAxes,ha="right",va="top",fontsize=8,
                color=C["teal"] if p_v<0.05 else C["gray"])
    ax.set_title(ylabel)

plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot14_framing_boxplot.png",bbox_inches="tight"); plt.close()
print("Saved plot14")

#PLOT 15: Valence (Positive/Negative/Control) vs groupthink & satisfaction — BOXPLOT
VALENCE_COLORS = {"Control":C["gray"],"Positive":C["teal"],"Negative":C["coral"]}
val_present = [v for v in VALENCE_ORDER if v in merged["valence"].values]

fig,axes=plt.subplots(1,4,figsize=(15,5))
fig.suptitle("Plot 15 (NEW) — Valence (Positive vs Negative AI framing) vs outcomes\n"
             "(Kruskal-Wallis; does positive/negative AI tone shift behaviour?)",
             fontsize=11,y=1.01)

for ax,(col,ylabel) in zip(axes,[
        ("groupthink_score","Groupthink /10"),
        ("satisfaction_score","Satisfaction /10"),
        ("stress_score","Stress /10"),
        ("regret","Regret (0/1)")]):
    data=[merged.loc[merged["valence"]==v,col].dropna().values for v in val_present]
    bp=ax.boxplot(data,patch_artist=True,
                  medianprops=dict(color="white",linewidth=2),
                  whiskerprops=dict(color="#555"),capprops=dict(color="#555"),
                  flierprops=dict(marker="",alpha=0))
    for patch,v in zip(bp["boxes"],val_present):
        patch.set_facecolor(VALENCE_COLORS[v]); patch.set_alpha(0.75)
    for i,(vals,v) in enumerate(zip(data,val_present)):
        ax.scatter(np.random.normal(i+1,0.07,len(vals)),vals,
                   color=VALENCE_COLORS[v],s=28,zorder=5,alpha=0.85)
    ax.set_xticklabels(val_present,fontsize=9); ax.set_xlabel("Valence"); ax.set_ylabel(ylabel)
    gs2=[d for d in data if len(d)>=2]
    if len(gs2)>=2:
        H_v,p_v=kruskal(*gs2)
        ax.text(0.97,0.97,f"KW p={p_v:.3f} {sig_label(p_v)}",
                transform=ax.transAxes,ha="right",va="top",fontsize=8,
                color=C["teal"] if p_v<0.05 else C["gray"])
    ax.set_title(ylabel)

plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot15_valence_boxplot.png",bbox_inches="tight"); plt.close()
print("Saved plot15")

#PLOT 16: Framing × Valence interaction dot plot (mean ± SE)
CONDITION_PAIRS = [
    ("Open Positive",    "Open",    "Positive"),
    ("Open Negative",    "Open",    "Negative"),
    ("Closed Positive",  "Closed",  "Positive"),
    ("Closed Negative",  "Closed",  "Negative"),
    ("Neutral Positive", "Neutral", "Positive"),
    ("Neutral Negative", "Neutral", "Negative"),
]
pair_present = [cp for cp in CONDITION_PAIRS if cp[0] in merged["condition_norm"].values]

metrics_interact = [
    ("ai_compliance_rate", "AI compliance rate"),
    ("groupthink_score",   "Groupthink /10"),
    ("satisfaction_score", "Satisfaction /10"),
    ("dissent_rate",       "Dissent rate"),
]

fig,axes=plt.subplots(1,4,figsize=(16,5))
fig.suptitle("Plot 16 (NEW) — Framing × Valence interaction (mean ± SE)\n"
             "Each dot = one condition; teal = Positive, coral = Negative",
             fontsize=11,y=1.01)

VALENCE_DOT = {"Positive":C["teal"],"Negative":C["coral"]}
FRAMING_MARKERS = {"Open":"o","Closed":"s","Neutral":"^"}

for ax,(col,ylabel) in zip(axes, metrics_interact):
    x_tick_labels = []
    for i,(cname,framing,valence) in enumerate(pair_present):
        vals = merged.loc[merged["condition_norm"]==cname, col].dropna()
        if len(vals) == 0: continue
        mn  = vals.mean()
        se  = vals.std() / np.sqrt(len(vals))
        ax.errorbar(i, mn, yerr=se,
                    fmt=FRAMING_MARKERS.get(framing,"o"),
                    color=VALENCE_DOT[valence],
                    markersize=9, capsize=4, lw=1.5, zorder=4)
        x_tick_labels.append(cname.replace(" ","\n"))

    # Connect same-framing pairs with lines
    framing_x = {}
    for i,(cname,framing,valence) in enumerate(pair_present):
        framing_x.setdefault(framing, []).append(
            (i, merged.loc[merged["condition_norm"]==cname, col].mean()))
    for framing, pts in framing_x.items():
        pts.sort()
        if len(pts)==2:
            ax.plot([pts[0][0],pts[1][0]],[pts[0][1],pts[1][1]],
                    color=C["gray"],lw=1,ls="--",alpha=0.5,zorder=2)

    ax.set_xticks(range(len(pair_present)))
    ax.set_xticklabels(x_tick_labels,fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(ylabel)

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0],marker="o",color=C["teal"], lw=0,markersize=8,label="Positive valence"),
    Line2D([0],[0],marker="o",color=C["coral"],lw=0,markersize=8,label="Negative valence"),
    Line2D([0],[0],marker="o",color=C["gray"], lw=0,markersize=8,label="Open framing"),
    Line2D([0],[0],marker="s",color=C["gray"], lw=0,markersize=8,label="Closed framing"),
    Line2D([0],[0],marker="^",color=C["gray"], lw=0,markersize=8,label="Neutral framing"),
]
axes[-1].legend(handles=legend_els, fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot16_condition_interaction_dotplot.png",bbox_inches="tight"); plt.close()
print("Saved plot16")

print(f"\nAll 16 plots saved to: {OUT_DIR}")

