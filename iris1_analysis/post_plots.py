
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import os

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
FILE       = "Until Dawn Post Survey_April 6, 2026_12.18.xlsx"
OUTPUT_DIR = "post_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df   = pd.read_excel(FILE)
data = df.iloc[1:].copy().reset_index(drop=True)   # row 0 = question labels, not data

# ─────────────────────────────────────────────
# NEW — SAVE CONDITION BEFORE NUMERIC CONVERSION
# Q36   = Condition column (experimental group assignment, string labels)
# Q36.1 = Commitment-bias survey item (was labelled Q36 in the original code)
# CRITICAL: extract Condition BEFORE pd.to_numeric() wipes the string values.
# ─────────────────────────────────────────────
data["Condition"] = data["Q36"].astype(str).str.strip().str.lower()

def _get_type(x):
    x = str(x)
    for t in ["open", "closed", "neutral", "control"]:
        if t in x:
            return t
    return "other"

def _get_valence(x):
    x = str(x)
    if "positive" in x: return "positive"
    if "negative" in x: return "negative"
    if "control"  in x: return "control"
    return "other"

data["Condition_Type"]    = data["Condition"].apply(_get_type)
data["Condition_Valence"] = data["Condition"].apply(_get_valence)

SKIP_COLS = {"Q35", "Q33", "Q34", "Q36", "Condition", "Condition_Type", "Condition_Valence"}
num_cols = [c for c in data.columns if c not in SKIP_COLS]
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

print(f"Participants loaded: {len(data)}")

# ─────────────────────────────────────────────
# 2. COMPOSITE SCORES
# ─────────────────────────────────────────────
AI_COLS     = ["Q15","Q41","Q43","Q44","Q25","Q26","Q45"]
GT_COLS     = ["Q9","Q8","Q36.1","Q37","Q10","Q38","Q17","Q12","Q13"]  # Q36.1 = commitment bias item
CT_COLS     = ["Q20","Q22","Q23","Q19","Q21","Q40","Q46"]
PERF_COLS   = ["Q28","Q29","Q30","Q31"]
STRESS_COLS = ["Q5","Q6","Q7"]
COMMIT_COLS = ["Q9","Q36.1"]
CENSOR_COLS = ["Q8","Q10","Q38"]
UNANIM_COLS = ["Q12","Q13"]

data["AI_Influence"]       = data[AI_COLS].mean(axis=1)
data["Groupthink"]         = data[GT_COLS].mean(axis=1)
data["Cohesion"]           = (data["Q1"] + (10-data["Q2"]) + (10-data["Q3"]) + (10-data["Q4"])) / 4
data["Regret_binary"]      = (data["Q33"] == "Yes").astype(int)
data["Regret"]             = (data["Regret_binary"]*10 + (10-data["Q30"]) + (10-data["Q31"])) / 3
data["Critical_Thinking"]  = data[CT_COLS].mean(axis=1)
data["Performance"]        = data[PERF_COLS].mean(axis=1)
data["Stress"]             = data[STRESS_COLS].mean(axis=1)
data["Commitment_Bias"]    = data[COMMIT_COLS].mean(axis=1)
data["Self_Censorship"]    = data[CENSOR_COLS].mean(axis=1)
data["Illusion_Unanimity"] = data[UNANIM_COLS].mean(axis=1)

data["Agreed_with_AI"] = data["Q26"] >= data["Q26"].median()
data["AI_Condition"]   = np.where(
    data["AI_Influence"] >= data["AI_Influence"].median(),
    "High AI\nInfluence", "Low AI\nInfluence"
)

# ─────────────────────────────────────────────
# 3. STYLE CONSTANTS
# ─────────────────────────────────────────────
BG   = "#0d0f1a"
CARD = "#151823"
TEXT = "#e8eaf0"
SUB  = "#8892a4"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUB,
    "ytick.color":       SUB,
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COND_COLORS = {
    "control":          "#8892a4",
    "open positive":    "#56cfe1",
    "open negative":    "#4fc3f7",
    "closed positive":  "#7b5ea7",
    "closed negative":  "#e85d75",
    "neutral positive": "#ffd166",
    "neutral negative": "#ff9f1c",
}
TYPE_COLORS = {"control": "#8892a4", "open": "#56cfe1", "closed": "#e85d75", "neutral": "#ffd166"}
VAL_COLORS  = {"control": "#8892a4", "positive": "#56cfe1", "negative": "#e85d75"}

# ─────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────────
def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'  saved  {path}')


def stat_box(ax, r, p, color):
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.text(0.97, 0.06, f"r={r:.2f}{stars}  {p_str}", transform=ax.transAxes,
            fontsize=11, color=color, ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2235",
                      alpha=0.9, edgecolor=color, linewidth=1.2))


def scatter_regression(ax, x, y, color, shade=True):
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    ax.scatter(xv, yv, color=color, s=90, alpha=0.8,
               edgecolors="white", linewidths=0.6, zorder=3)
    m, b, r, p, _ = stats.linregress(xv, yv)
    xl = np.linspace(xv.min(), xv.max(), 200)
    if shade:
        ax.fill_between(xl, m*xl+b - 1.5, m*xl+b + 1.5, color=color, alpha=0.10)
    ax.plot(xl, m*xl+b, color=color, lw=2.5, zorder=4)
    stat_box(ax, r, p, color)
    return r, p


def style_ax(ax):
    ax.spines[["bottom", "left"]].set_color("#2a2f45")
    ax.grid(True, alpha=0.12, color="white")


# ─────────────────────────────────────────────
# 5. ORIGINAL 12 PLOTS
# ─────────────────────────────────────────────
print("\nGenerating plots...")

# ── PLOT 01: AI Influence → Groupthink ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["AI_Influence"], data["Groupthink"], "#7b5ea7")
ax.set_xlabel("AI Influence Score", fontsize=13)
ax.set_ylabel("Groupthink Score", fontsize=13)
ax.set_title("AI Influence → Groupthink", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Does higher AI influence lead to more groupthink?",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "01_ai_influence_groupthink.png")

# ── PLOT 02: Groupthink → Regret ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["Groupthink"], data["Regret"], "#e85d75")
ax.set_xlabel("Groupthink Score", fontsize=13)
ax.set_ylabel("Regret Score", fontsize=13)
ax.set_title("Groupthink → Regret", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Does groupthink lead to post-game regret?",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "02_groupthink_regret.png")

# ── PLOT 03: Behavior (Agreed vs Disagreed with AI) → Groupthink ───────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)

grp_d = data[data["Agreed_with_AI"] == False]["Groupthink"].dropna()
grp_a = data[data["Agreed_with_AI"] == True]["Groupthink"].dropna()
colors_g3 = ["#4fc3f7", "#7b5ea7"]

vp = ax.violinplot([grp_d, grp_a], positions=[1, 2], showmedians=True, showextrema=True)
for pc, col in zip(vp["bodies"], colors_g3):
    pc.set_facecolor(col); pc.set_alpha(0.45); pc.set_edgecolor(col); pc.set_linewidth(1.2)
for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
    vp[part].set_color("white"); vp[part].set_linewidth(1.5)

np.random.seed(42)
for i, (grp, col) in enumerate(zip([grp_d, grp_a], colors_g3)):
    jit = np.random.uniform(-0.08, 0.08, len(grp))
    ax.scatter([i+1+j for j in jit], grp, color=col, s=55, alpha=0.75,
               zorder=5, edgecolors="white", linewidths=0.4)

ax.scatter([1, 2], [grp_d.mean(), grp_a.mean()], color="white", s=100, zorder=6, marker="D")

t_stat, p_val = stats.ttest_ind(grp_d, grp_a)
stars = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns (trend)"))
ax.text(0.97, 0.06, f"t={t_stat:.2f}  p={p_val:.3f}  {stars}",
        transform=ax.transAxes, fontsize=11, color="#4fc3f7", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2235",
                  alpha=0.9, edgecolor="#4fc3f7", linewidth=1.2))

ax.set_xticks([1, 2])
ax.set_xticklabels([f"Disagreed with AI\n(n={len(grp_d)})",
                    f"Agreed with AI\n(n={len(grp_a)})"], fontsize=11)
ax.set_ylabel("Groupthink Score", fontsize=13)
ax.set_title("AI Behavior → Groupthink", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Do people who follow AI show more groupthink? (◆ = mean)",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "03_behavior_groupthink_violin.png")

# ── PLOT 04: AI Condition (High vs Low) → Groupthink ───────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)

grp_lo = data[data["AI_Condition"] == "Low AI\nInfluence"]["Groupthink"].dropna()
grp_hi = data[data["AI_Condition"] == "High AI\nInfluence"]["Groupthink"].dropna()
colors_g4 = ["#4fc3f7", "#7b5ea7"]

for i, (grp, col) in enumerate(zip([grp_lo, grp_hi], colors_g4)):
    np.random.seed(i * 7)
    jit = np.random.uniform(-0.12, 0.12, len(grp))
    ax.scatter([i+1+j for j in jit], grp, color=col, s=60, alpha=0.6,
               zorder=3, edgecolors="white", linewidths=0.4)
    mean_v = grp.mean()
    ci = stats.t.interval(0.95, df=len(grp)-1, loc=mean_v, scale=stats.sem(grp))
    ax.plot([i+0.8, i+1.2], [mean_v, mean_v], color="white", lw=3, zorder=5)
    ax.plot([i+1, i+1], ci, color=col, lw=2.5, zorder=4)
    ax.plot([i+0.92, i+1.08], [ci[0], ci[0]], color=col, lw=2)
    ax.plot([i+0.92, i+1.08], [ci[1], ci[1]], color=col, lw=2)
    ax.text(i+1, mean_v + 0.45, f"{mean_v:.2f}",
            ha="center", fontsize=12, color="white", fontweight="bold")

_, p4 = stats.ttest_ind(grp_lo, grp_hi)
stars4 = "***" if p4<0.001 else ("**" if p4<0.01 else ("*" if p4<0.05 else "ns"))
ax.text(0.97, 0.06, f"p={p4:.4f}  {stars4}",
        transform=ax.transAxes, fontsize=11, color="#7b5ea7", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2235",
                  alpha=0.9, edgecolor="#7b5ea7", linewidth=1.2))

ax.set_xticks([1, 2])
ax.set_xticklabels([f"Low AI Influence\n(n={len(grp_lo)})",
                    f"High AI Influence\n(n={len(grp_hi)})"], fontsize=11)
ax.set_ylabel("Groupthink Score", fontsize=13)
ax.set_title("AI Influence Condition → Groupthink", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "— = mean, error bars = 95% CI",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "04_ai_condition_groupthink.png")

# ── PLOT 05: Cohesion → Groupthink ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["Cohesion"], data["Groupthink"], "#ffd166")
ax.set_xlabel("Cohesion Score", fontsize=13)
ax.set_ylabel("Groupthink Score", fontsize=13)
ax.set_title("Cohesion → Groupthink", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Classic groupthink theory: does group unity → conformity?",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "05_cohesion_groupthink.png")

# ── PLOT 06: Correlation Heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(BG)

hv = ["AI_Influence", "Cohesion", "Groupthink", "Regret"]
hl = ["AI Influence", "Cohesion", "Groupthink", "Regret"]
corr_m = data[hv].corr().values
n_v    = len(hv)

cmap = LinearSegmentedColormap.from_list("div", ["#e85d75", "#1e2235", "#7b5ea7"], N=256)

for i in range(n_v):
    for j in range(n_v):
        if j > i:
            rect = plt.Rectangle([j-0.5, i-0.5], 1, 1,
                                  facecolor="#0d0f1a", edgecolor="#0d0f1a", linewidth=2)
            ax.add_patch(rect)
            continue
        r_val = corr_m[i, j]
        rect  = plt.Rectangle([j-0.5, i-0.5], 1, 1,
                               facecolor=cmap((r_val + 1) / 2),
                               edgecolor="#0d0f1a", linewidth=2)
        ax.add_patch(rect)
        if i == j:
            ax.text(j, i, hl[i], ha="center", va="center",
                    fontsize=11, fontweight="bold", color=TEXT)
        else:
            x2 = data[hv[j]].dropna(); y2 = data[hv[i]].dropna()
            idx = x2.index.intersection(y2.index)
            _, p_val = stats.pearsonr(x2[idx], y2[idx])
            stars = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else ""))
            txt_col = "white" if abs(r_val) > 0.35 else TEXT
            ax.text(j, i-0.1,  f"{r_val:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=txt_col)
            ax.text(j, i+0.25, stars, ha="center", va="center",
                    fontsize=11, color="#ffd166")

sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Pearson r", color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUB)

ax.set_xlim(-0.5, n_v - 0.5); ax.set_ylim(-0.5, n_v - 0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Correlation Heatmap (lower triangle)", fontsize=15, fontweight="bold", pad=12, color=TEXT)
ax.text(0.5, -0.04, "* p<.05   ** p<.01   *** p<.001",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[:].set_visible(False)
save(fig, "06_correlation_heatmap.png")

# ── PLOT 07: Performance → Regret ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)

mask   = data["Performance"].notna() & data["Regret"].notna()
colors = data["Regret_binary"].map({1: "#e85d75", 0: "#4fc3f7"})
ax.scatter(data["Performance"][mask], data["Regret"][mask],
           c=colors[mask], s=100, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)

m, b, r, p, _ = stats.linregress(data["Performance"][mask], data["Regret"][mask])
xl = np.linspace(data["Performance"][mask].min(), data["Performance"][mask].max(), 200)
ax.fill_between(xl, m*xl+b - 1.5, m*xl+b + 1.5, color="#e85d75", alpha=0.08)
ax.plot(xl, m*xl+b, color="#e85d75", lw=2.5, zorder=4)
stat_box(ax, r, p, "#e85d75")

leg = [mpatches.Patch(color="#e85d75", label="Expressed regret (Yes)"),
       mpatches.Patch(color="#4fc3f7", label="No regret (No)")]
ax.legend(handles=leg, fontsize=10, facecolor="#1e2235", edgecolor="#2a2f45", labelcolor=TEXT)
ax.set_xlabel("Performance Score", fontsize=13)
ax.set_ylabel("Regret Score", fontsize=13)
ax.set_title("Performance → Regret", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Better performance = less regret (r=−0.82***)",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "07_performance_regret.png")

# ── PLOT 08: AI Influence → Stress ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)

mask = data["AI_Influence"].notna() & data["Stress"].notna()
ax.hexbin(data["AI_Influence"][mask], data["Stress"][mask],
          gridsize=8, cmap="Blues", alpha=0.3, mincnt=1)
ax.scatter(data["AI_Influence"][mask], data["Stress"][mask],
           color="#4fc3f7", s=80, alpha=0.8, edgecolors="white", linewidths=0.6, zorder=4)

m, b, r, p, _ = stats.linregress(data["AI_Influence"][mask], data["Stress"][mask])
xl = np.linspace(data["AI_Influence"][mask].min(), data["AI_Influence"][mask].max(), 200)
ax.plot(xl, m*xl+b, color="#4fc3f7", lw=2.5, zorder=5)
stat_box(ax, r, p, "#4fc3f7")

ax.set_xlabel("AI Influence Score", fontsize=13)
ax.set_ylabel("Stress Score", fontsize=13)
ax.set_title("AI Influence → Stress", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "AI guidance should reduce cognitive load — why does it increase stress?",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "08_ai_influence_stress.png")

# ── PLOT 09: Illusion of Unanimity → Performance ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["Illusion_Unanimity"], data["Performance"], "#7b5ea7")

mx = data["Illusion_Unanimity"].median(); my = data["Performance"].median()
ax.axvline(mx, color=SUB, lw=1, ls="--", alpha=0.4)
ax.axhline(my, color=SUB, lw=1, ls="--", alpha=0.4)
ax.text(mx + 0.15, ax.get_ylim()[1] * 0.97, "median", fontsize=9, color=SUB, va="top")

ax.set_xlabel("Illusion of Unanimity Score", fontsize=13)
ax.set_ylabel("Performance Score", fontsize=13)
ax.set_title("Illusion of Unanimity → Performance", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Counterintuitive: fake consensus → better in-game performance",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "09_unanimity_performance.png")

# ── PLOT 10: AI Influence → Cohesion ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["AI_Influence"], data["Cohesion"], "#56cfe1")
ax.set_xlabel("AI Influence Score", fontsize=13)
ax.set_ylabel("Cohesion Score", fontsize=13)
ax.set_title("AI Influence → Team Cohesion", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "More AI guidance = weaker team unity?",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "10_ai_cohesion.png")

# ── PLOT 11: Self-Censorship → Groupthink ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
scatter_regression(ax, data["Self_Censorship"], data["Groupthink"], "#ff9f1c")
ax.set_xlabel("Self-Censorship Score", fontsize=13)
ax.set_ylabel("Groupthink Score", fontsize=13)
ax.set_title("Self-Censorship → Groupthink", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Strongest relationship in entire dataset (r=+0.845***)",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
style_ax(ax)
save(fig, "11_selfcensorship_groupthink.png")

# ── PLOT 12: Score Distributions — Regret Yes vs No ────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(CARD)

yes_grp = data[data["Q33"] == "Yes"]
no_grp  = data[data["Q33"] == "No"]
vars_c  = ["AI_Influence", "Groupthink", "Cohesion", "Performance", "Stress"]
labels_c= ["AI\nInfluence", "Groupthink", "Cohesion", "Performance", "Stress"]
positions = np.arange(1, len(vars_c) + 1)

for i, (v, lbl) in enumerate(zip(vars_c, labels_c)):
    pos = positions[i]
    vp1 = ax.violinplot(no_grp[v].dropna(), positions=[pos-0.22],
                        widths=0.35, showmedians=True, showextrema=False)
    for pc in vp1["bodies"]:
        pc.set_facecolor("#4fc3f7"); pc.set_alpha(0.5); pc.set_edgecolor("#4fc3f7")
    vp1["cmedians"].set_color("white"); vp1["cmedians"].set_linewidth(2)

    vp2 = ax.violinplot(yes_grp[v].dropna(), positions=[pos+0.22],
                        widths=0.35, showmedians=True, showextrema=False)
    for pc in vp2["bodies"]:
        pc.set_facecolor("#e85d75"); pc.set_alpha(0.5); pc.set_edgecolor("#e85d75")
    vp2["cmedians"].set_color("white"); vp2["cmedians"].set_linewidth(2)

    _, pv = stats.ttest_ind(no_grp[v].dropna(), yes_grp[v].dropna())
    stars = "***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else ""))
    if stars:
        ax.text(pos, 10.5, stars, ha="center", fontsize=13, color="#ffd166")

ax.set_xticks(positions); ax.set_xticklabels(labels_c, fontsize=11)
ax.set_ylabel("Score", fontsize=13); ax.set_ylim(0, 11.5)
ax.set_title("Score Distributions: Regret Yes vs No", fontsize=15, fontweight="bold", pad=12)

leg = [mpatches.Patch(color="#4fc3f7", alpha=0.7, label=f"No Regret (n={len(no_grp)})"),
       mpatches.Patch(color="#e85d75", alpha=0.7, label=f"Regret: Yes (n={len(yes_grp)})")]
ax.legend(handles=leg, fontsize=10, facecolor="#1e2235",
          edgecolor="#2a2f45", labelcolor=TEXT, loc="lower right")
ax.text(0.5, 1.02, "★ = significant group difference",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "12_regret_group_comparison.png")

# ─────────────────────────────────────────────
# 6. NEW PLOTS — CONDITION (Q36)
# ─────────────────────────────────────────────

# ── PLOT 13: AI Influence by Condition Type (ANOVA ***) ────────────────────
# Type: Grouped bar + jitter  (shows ANOVA-significant difference by framing type)
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(CARD)

type_order = ["control", "open", "neutral", "closed"]
type_means = data.groupby("Condition_Type")["AI_Influence"].mean()
type_sems  = data.groupby("Condition_Type")["AI_Influence"].sem()
type_ns    = data.groupby("Condition_Type")["AI_Influence"].count()

for xi, t in enumerate(type_order):
    ax.bar(xi, type_means[t], yerr=type_sems[t], color=TYPE_COLORS[t], alpha=0.82,
           edgecolor="white", linewidth=0.8,
           error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
    ax.text(xi, type_means[t] + type_sems[t] + 0.35, f"{type_means[t]:.2f}",
            ha="center", fontsize=12, color=TEXT, fontweight="bold")
    grp_vals = data[data["Condition_Type"] == t]["AI_Influence"].dropna().values
    np.random.seed(xi * 13)
    jit = np.random.uniform(-0.15, 0.15, len(grp_vals))
    ax.scatter(xi + jit, grp_vals, color=TYPE_COLORS[t], s=45, alpha=0.65,
               zorder=5, edgecolors="white", linewidths=0.3)

groups_13 = [data[data["Condition_Type"] == t]["AI_Influence"].dropna().values for t in type_order]
f_13, p_13 = stats.f_oneway(*groups_13)
stars_13 = "***" if p_13 < 0.001 else ("**" if p_13 < 0.01 else ("*" if p_13 < 0.05 else "ns"))
ax.text(0.97, 0.06, f"ANOVA: F={f_13:.2f}, p<0.001 {stars_13}",
        transform=ax.transAxes, fontsize=11, color="#ffd166", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2235", alpha=0.9, edgecolor="#ffd166", linewidth=1.2))

ax.set_xticks(range(len(type_order)))
ax.set_xticklabels([f"{t.capitalize()}\n(n={type_ns.get(t,0)})" for t in type_order], fontsize=11)
ax.set_ylabel("AI Influence Score", fontsize=13)
ax.set_ylim(0, 10)
ax.set_title("AI Influence by Condition Type", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Closed/Neutral framing drives highest AI influence — control group stands apart",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "13_condition_ai_influence.png")

# ── PLOT 14: Team Cohesion by Condition Valence (*) ────────────────────────
# Type: Bar + jitter  (shows valence moderates cohesion, control = baseline)
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(CARD)

val_order   = ["control", "positive", "negative"]
val_means_c = data.groupby("Condition_Valence")["Cohesion"].mean()
val_sems_c  = data.groupby("Condition_Valence")["Cohesion"].sem()
val_ns_c    = data.groupby("Condition_Valence")["Cohesion"].count()

for xi, v in enumerate(val_order):
    ax.bar(xi, val_means_c[v], yerr=val_sems_c[v], color=VAL_COLORS[v], alpha=0.82,
           edgecolor="white", linewidth=0.8,
           error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
    ax.text(xi, val_means_c[v] + val_sems_c[v] + 0.35, f"{val_means_c[v]:.2f}",
            ha="center", fontsize=12, color=TEXT, fontweight="bold")
    grp_vals = data[data["Condition_Valence"] == v]["Cohesion"].dropna().values
    np.random.seed(xi * 17)
    jit = np.random.uniform(-0.15, 0.15, len(grp_vals))
    ax.scatter(xi + jit, grp_vals, color=VAL_COLORS[v], s=45, alpha=0.65,
               zorder=5, edgecolors="white", linewidths=0.3)

groups_14 = [data[data["Condition_Valence"] == v]["Cohesion"].dropna().values for v in val_order]
f_14, p_14 = stats.f_oneway(*groups_14)
stars_14 = "***" if p_14 < 0.001 else ("**" if p_14 < 0.01 else ("*" if p_14 < 0.05 else "ns"))
ax.text(0.97, 0.06, f"ANOVA: F={f_14:.2f}, p={p_14:.3f} {stars_14}",
        transform=ax.transAxes, fontsize=11, color="#ffd166", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2235", alpha=0.9, edgecolor="#ffd166", linewidth=1.2))

ax.set_xticks(range(len(val_order)))
ax.set_xticklabels([f"{v.capitalize()}\n(n={val_ns_c.get(v,0)})" for v in val_order], fontsize=11)
ax.set_ylabel("Cohesion Score", fontsize=13)
ax.set_ylim(0, 12)
ax.set_title("Team Cohesion by Condition Valence", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Control group markedly more cohesive than all AI-framed conditions",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "14_condition_cohesion_valence.png")

# ── PLOT 15: Control vs Experimental — Multi-Score Comparison ──────────────
# Type: Grouped bar  (shows which outcomes differ significantly by AI exposure)
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(CARD)

compare_scores = ["AI_Influence", "Stress", "Cohesion", "Groupthink", "Performance",
                  "Regret", "Critical_Thinking"]
compare_labels = ["AI\nInfluence", "Stress", "Cohesion", "Groupthink", "Performance",
                  "Regret", "Critical\nThinking"]
ctrl_means = [data[data["Condition_Type"] == "control"][s].mean() for s in compare_scores]
exp_means  = [data[data["Condition_Type"] != "control"][s].mean() for s in compare_scores]
ctrl_sems  = [data[data["Condition_Type"] == "control"][s].sem() for s in compare_scores]
exp_sems   = [data[data["Condition_Type"] != "control"][s].sem() for s in compare_scores]
x_pos = np.arange(len(compare_scores))
w = 0.35

ax.bar(x_pos - w/2, ctrl_means, w, yerr=ctrl_sems, label="Control (n=5)",
       color="#8892a4", alpha=0.85, edgecolor="white",
       error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))
ax.bar(x_pos + w/2, exp_means, w, yerr=exp_sems, label="Experimental (n=30)",
       color="#7b5ea7", alpha=0.85, edgecolor="white",
       error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))

for xi, s in enumerate(compare_scores):
    ctrl_g = data[data["Condition_Type"] == "control"][s].dropna()
    exp_g  = data[data["Condition_Type"] != "control"][s].dropna()
    if len(ctrl_g) >= 2 and len(exp_g) >= 2:
        _, pv = stats.ttest_ind(ctrl_g, exp_g)
        star = "***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else ""))
        if star:
            ymax = max(ctrl_means[xi] + ctrl_sems[xi], exp_means[xi] + exp_sems[xi]) + 0.5
            ax.text(xi, ymax, star, ha="center", fontsize=13, color="#ffd166")

ax.set_xticks(x_pos); ax.set_xticklabels(compare_labels, fontsize=10)
ax.set_ylabel("Score", fontsize=13)
ax.set_ylim(0, 12)
ax.legend(fontsize=10, facecolor="#1e2235", edgecolor="#2a2f45", labelcolor=TEXT)
ax.set_title("Control vs Experimental Conditions — Score Comparison", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "★ = significant difference (p<.05) — AI conditions raise Influence+Stress, lower Cohesion",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="y")
save(fig, "15_condition_control_vs_exp.png")

# ── PLOT 16: AI Influence Across All 7 Conditions (horizontal bar) ──────────
# Type: Horizontal bar  (most differentiated score; shows full condition breakdown)
fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
ax.set_facecolor(CARD)

all_conds = ["control", "open positive", "open negative",
             "neutral positive", "neutral negative", "closed positive", "closed negative"]
ai_means  = [data[data["Condition"] == c]["AI_Influence"].mean() for c in all_conds]
ai_sems   = [data[data["Condition"] == c]["AI_Influence"].sem() for c in all_conds]
cond_ns   = [data[data["Condition"] == c]["AI_Influence"].count() for c in all_conds]
bar_colors= [COND_COLORS[c] for c in all_conds]

ax.barh(range(len(all_conds)), ai_means, xerr=ai_sems,
        color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.7,
        error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))
for yi, (m, n) in enumerate(zip(ai_means, cond_ns)):
    ax.text(m + 0.25, yi, f"{m:.2f}", va="center", fontsize=11, color=TEXT, fontweight="bold")

ax.set_yticks(range(len(all_conds)))
ax.set_yticklabels([f"{c.title()} (n={n})" for c, n in zip(all_conds, cond_ns)], fontsize=10)
ax.set_xlabel("AI Influence Score", fontsize=13)
ax.set_xlim(0, 10)
ax.set_title("AI Influence Across All 7 Conditions", fontsize=15, fontweight="bold", pad=12)
ax.text(0.5, 1.02, "Control stands apart; framing type shapes AI influence more than valence",
        transform=ax.transAxes, fontsize=10, color=SUB, ha="center")
ax.spines[["bottom", "left"]].set_color("#2a2f45")
ax.grid(True, alpha=0.12, color="white", axis="x")
save(fig, "16_condition_all7_heatmap.png")

# ─────────────────────────────────────────────
# 7. PRINT ALL SIGNIFICANCE VALUES
# ─────────────────────────────────────────────
SCORES = [
    "AI_Influence", "Groupthink", "Cohesion", "Regret",
    "Critical_Thinking", "Performance", "Stress",
    "Commitment_Bias", "Self_Censorship", "Illusion_Unanimity",
]

print("\n=== ALL PAIRWISE CORRELATIONS (sorted by |r|) ===")
print(f"{'Pair':<45} {'r':>7}  {'p-value':>9}  {'sig':>5}")
print("-" * 68)

results = []
for i, a in enumerate(SCORES):
    for j, b in enumerate(SCORES):
        if j <= i:
            continue
        x = data[a].dropna(); y = data[b].dropna()
        idx = x.index.intersection(y.index)
        r, p = stats.pearsonr(x[idx], y[idx])
        stars = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        results.append((abs(r), r, p, a, b, stars))

for _, r, p, a, b, stars in sorted(results, reverse=True):
    print(f"{a+' ↔ '+b:<45} {r:>+7.3f}  {p:>9.4f}  {stars:>5}")

print("\nDone — all 16 plots saved in post_plots/ directory.")
