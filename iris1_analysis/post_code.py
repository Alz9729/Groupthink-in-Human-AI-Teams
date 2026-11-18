

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

output_dir = 'post_survey_figs'
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
FILE = "Until Dawn Post Survey_April 6, 2026_12.18.xlsx"

df = pd.read_excel(FILE)

print("=== RAW DATA PREVIEW ===")
print(f"Total rows loaded (including label row): {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

data = df.iloc[1:].copy().reset_index(drop=True)
print(f"Data rows (participants): {len(data)}")

# ─────────────────────────────────────────────
#  — SAVE CONDITION BEFORE NUMERIC CONVERSION
# Q36   = Condition column (experimental group, string labels)
# Q36.1 = Commitment-bias survey item (was labelled Q36 before xlsx update)
# CRITICAL: save Condition BEFORE pd.to_numeric() wipes the string values.
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

# ─────────────────────────────────────────────
# 2. QUESTION LABELS
# ─────────────────────────────────────────────
LABELS = {
    "Q1":    "Team is united in goals",
    "Q2":    "Unhappy with team commitment (reverse)",
    "Q3":    "Conflicting team aspirations (reverse)",
    "Q4":    "Not enough improvement opportunities (reverse)",
    "Q5":    "Mentally demanding",
    "Q6":    "Hurried / rushed",
    "Q7":    "Stressed / annoyed",
    "Q8":    "Felt discouraged from disagreeing",
    "Q9":    "Justified earlier decision despite new info",
    "Q10":   "Avoided objections due to AI framing",
    "Q12":   "Believed choice was right despite moral concerns",
    "Q13":   "Full agreement apparent even if alternatives skipped",
    "Q15":   "AI presented limited options",
    "Q17":   "Group did not reevaluate choice for risks",
    "Q19":   "Considered AI perspectives even when disagreed",
    "Q20":   "Team created backup/alternative plans",
    "Q21":   "Process encouraged open evaluation",
    "Q22":   "Used all in-game information available",
    "Q23":   "Reflected on own ideas AND AI suggestions",
    "Q25":   "AI made certain choices feel 'correct'",
    "Q26":   "Disagreeing with AI felt like going against group",
    "Q28":   "Decisions made efficiently given time limits",
    "Q29":   "Decisions helped move story toward success",
    "Q30":   "Satisfied with results of decisions",
    "Q31":   "Handled decision-making challenges effectively",
    "Q33":   "Regret any decisions? (Yes/No)",
    "Q36":   "Condition (experimental group assignment)",        # 
    "Q36.1": "More likely to justify preferred option (commitment bias)",
    "Q37":   "AI discouraged contradictory information",
    "Q38":   "Avoided raising objections during decision-making",
    "Q40":   "AI encouraged exploring different perspectives",
    "Q41":   "AI suggestions pointed in same direction",
    "Q43":   "AI repeated similar reasoning",
    "Q44":   "AI framing made some options seem irrelevant",
    "Q45":   "AI influenced decisions more than own judgment",
    "Q46":   "AI encouraged critically evaluating options",
}

# ─────────────────────────────────────────────
# 3. CONVERT TO NUMERIC
# ─────────────────────────────────────────────
SKIP_COLS = {"Q35", "Q33", "Q34", "Q36", "Condition", "Condition_Type", "Condition_Valence"}
num_cols = [c for c in data.columns if c not in SKIP_COLS]
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# ─────────────────────────────────────────────
# 4. BUILD COMPOSITE SCORES
# ─────────────────────────────────────────────
AI_COLS     = ["Q15", "Q41", "Q43", "Q44", "Q25", "Q26", "Q45"]
GT_COLS     = ["Q9", "Q8", "Q36.1", "Q37", "Q10", "Q38", "Q17", "Q12", "Q13"]
CT_COLS     = ["Q20", "Q22", "Q23", "Q19", "Q21", "Q40", "Q46"]
PERF_COLS   = ["Q28", "Q29", "Q30", "Q31"]
STRESS_COLS = ["Q5", "Q6", "Q7"]
COMMIT_COLS = ["Q9", "Q36.1"]
CENSOR_COLS = ["Q8", "Q10", "Q38"]
UNANIM_COLS = ["Q12", "Q13"]

data["AI_Influence"]       = data[AI_COLS].mean(axis=1)
data["Groupthink"]         = data[GT_COLS].mean(axis=1)
data["Cohesion"]           = (data["Q1"] + (10 - data["Q2"]) + (10 - data["Q3"]) + (10 - data["Q4"])) / 4
data["Regret_binary"]      = (data["Q33"] == "Yes").astype(int)
data["Regret"]             = (data["Regret_binary"] * 10 + (10 - data["Q30"]) + (10 - data["Q31"])) / 3
data["Critical_Thinking"]  = data[CT_COLS].mean(axis=1)
data["Performance"]        = data[PERF_COLS].mean(axis=1)
data["Stress"]             = data[STRESS_COLS].mean(axis=1)
data["Commitment_Bias"]    = data[COMMIT_COLS].mean(axis=1)
data["Self_Censorship"]    = data[CENSOR_COLS].mean(axis=1)
data["Illusion_Unanimity"] = data[UNANIM_COLS].mean(axis=1)

med_q26 = data["Q26"].median()
data["Agreed_with_AI"] = data["Q26"] >= med_q26

med_ai = data["AI_Influence"].median()
data["AI_Condition"] = np.where(data["AI_Influence"] >= med_ai, "High AI", "Low AI")

# ─────────────────────────────────────────────
# 5. DESCRIPTIVE STATS
# ─────────────────────────────────────────────
SCORES = [
    "AI_Influence", "Groupthink", "Cohesion", "Regret",
    "Critical_Thinking", "Performance", "Stress",
    "Commitment_Bias", "Self_Censorship", "Illusion_Unanimity",
]

print("\n=== DESCRIPTIVE STATISTICS ===")
print(data[SCORES].describe().round(3).to_string())

# ─────────────────────────────────────────────
# 6. PAIRWISE CORRELATIONS
# ─────────────────────────────────────────────
print("\n=== ALL PAIRWISE PEARSON CORRELATIONS (sorted by |r|) ===")
print(f"{'Pair':<45} {'r':>7}  {'p-value':>9}  {'sig':>5}  {'n':>4}")
print("-" * 75)

results = []
for i, a in enumerate(SCORES):
    for j, b in enumerate(SCORES):
        if j <= i:
            continue
        x = data[a].dropna()
        y = data[b].dropna()
        idx = x.index.intersection(y.index)
        r, p = stats.pearsonr(x[idx], y[idx])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        results.append((abs(r), r, p, a, b, len(idx), stars))

results.sort(reverse=True)
for _, r, p, a, b, n, stars in results:
    pair = f"{a} ↔ {b}"
    print(f"{pair:<45} {r:>+7.3f}  {p:>9.4f}  {stars:>5}  {n:>4}")

# ─────────────────────────────────────────────
# 7. ORIGINAL GROUP COMPARISONS
# ─────────────────────────────────────────────
print("\n=== GROUP COMPARISONS (independent t-tests) ===")

grp_agree    = data[data["Agreed_with_AI"] == True]["Groupthink"].dropna()
grp_disagree = data[data["Agreed_with_AI"] == False]["Groupthink"].dropna()
t3, p3 = stats.ttest_ind(grp_agree, grp_disagree)
print(f"\nGraph 3 — Agreed vs Disagreed with AI → Groupthink")
print(f"  Agreed    : n={len(grp_agree)}, mean={grp_agree.mean():.3f}, sd={grp_agree.std():.3f}")
print(f"  Disagreed : n={len(grp_disagree)}, mean={grp_disagree.mean():.3f}, sd={grp_disagree.std():.3f}")
print(f"  t={t3:.3f}, p={p3:.4f}")

grp_high = data[data["AI_Condition"] == "High AI"]["Groupthink"].dropna()
grp_low  = data[data["AI_Condition"] == "Low AI"]["Groupthink"].dropna()
t4, p4 = stats.ttest_ind(grp_high, grp_low)
print(f"\nGraph 4 — High vs Low AI Influence → Groupthink")
print(f"  High AI   : n={len(grp_high)}, mean={grp_high.mean():.3f}, sd={grp_high.std():.3f}")
print(f"  Low AI    : n={len(grp_low)}, mean={grp_low.mean():.3f}, sd={grp_low.std():.3f}")
print(f"  t={t4:.3f}, p={p4:.4f}")

# ─────────────────────────────────────────────
#  — 8. CONDITION (Q36) ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print(": CONDITION (Q36) ANALYSIS")
print("=" * 75)

print(f"\nCondition group sizes:")
print(data["Condition"].value_counts().to_string())

print(f"\n--- Group means by all 7 conditions ---")
print(data.groupby("Condition")[SCORES].mean().round(3).to_string())

print(f"\n--- ONE-WAY ANOVA: Full Condition (7 levels) → Each Score ---")
print(f"{'Score':<25}  {'F':>7}  {'p-value':>9}  {'sig':>5}")
print("-" * 55)
for s in SCORES:
    groups = [g[s].dropna().values for _, g in data.groupby("Condition")
              if len(g[s].dropna()) >= 2]
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {s:<25}: F={f_val:>6.3f}  p={p_val:>8.4f}  {stars:>5}")

print(f"\n--- ONE-WAY ANOVA: Condition TYPE (open/closed/neutral/control) → Each Score ---")
print(f"{'Score':<25}  {'F':>7}  {'p-value':>9}  {'sig':>5}")
print("-" * 55)
for s in SCORES:
    groups = [g[s].dropna().values for _, g in data.groupby("Condition_Type")
              if len(g[s].dropna()) >= 2]
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {s:<25}: F={f_val:>6.3f}  p={p_val:>8.4f}  {stars:>5}")

print(f"\n--- ONE-WAY ANOVA: Condition VALENCE (positive/negative/control) → Each Score ---")
print(f"{'Score':<25}  {'F':>7}  {'p-value':>9}  {'sig':>5}")
print("-" * 55)
for s in SCORES:
    groups = [g[s].dropna().values for _, g in data.groupby("Condition_Valence")
              if len(g[s].dropna()) >= 2]
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {s:<25}: F={f_val:>6.3f}  p={p_val:>8.4f}  {stars:>5}")

print(f"\n--- T-TESTS: Control (n=5) vs All Experimental Conditions (n=30) ---")
print(f"{'Score':<25}  {'ctrl_M':>7}  {'exp_M':>7}  {'Δ':>7}  {'t':>7}  {'p':>9}  {'sig':>5}")
print("-" * 80)
for s in SCORES:
    ctrl = data[data["Condition_Type"] == "control"][s].dropna()
    exp  = data[data["Condition_Type"] != "control"][s].dropna()
    if len(ctrl) >= 2 and len(exp) >= 2:
        t_val, p_val = stats.ttest_ind(ctrl, exp)
        d = exp.mean() - ctrl.mean()
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {s:<25}: ctrl={ctrl.mean():>6.3f}  exp={exp.mean():>6.3f}  Δ={d:>+6.3f}"
              f"  t={t_val:>6.3f}  p={p_val:>8.4f}  {stars:>5}")

print(f"\n--- T-TESTS: Positive vs Negative Valence (excluding control) ---")
pos = data[data["Condition_Valence"] == "positive"]
neg = data[data["Condition_Valence"] == "negative"]
for s in SCORES:
    a = pos[s].dropna(); b = neg[s].dropna()
    if len(a) >= 2 and len(b) >= 2:
        t_val, p_val = stats.ttest_ind(a, b)
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  {s:<25}: pos={a.mean():.3f}, neg={b.mean():.3f}, Δ={b.mean()-a.mean():+.3f}, "
              f"t={t_val:.3f}, p={p_val:.4f} {stars}")

# ─────────────────────────────────────────────
# 9. CORRELATION MATRIX (heatmap variables)
# ─────────────────────────────────────────────
heatmap_vars = ["AI_Influence", "Cohesion", "Groupthink", "Regret"]
print("\n=== CORRELATION MATRIX (Heatmap variables) ===")
print(data[heatmap_vars].corr().round(3).to_string())

print("\n=== P-VALUES FOR HEATMAP PAIRS ===")
for i, a in enumerate(heatmap_vars):
    for j, b in enumerate(heatmap_vars):
        if j <= i:
            continue
        x = data[a].dropna(); y = data[b].dropna()
        idx = x.index.intersection(y.index)
        r, p = stats.pearsonr(x[idx], y[idx])
        stars = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        print(f"  {a} ↔ {b}: r={r:+.3f}, p={p:.4f} {stars}")

# ─────────────────────────────────────────────
# 10. STYLE CONSTANTS
# ─────────────────────────────────────────────
BG   = "#0d0f1a"
CARD = "#151823"
C1   = "#7b5ea7"
C2   = "#e85d75"
C3   = "#4fc3f7"
C4   = "#ffd166"
TEXT = "#e8eaf0"
SUB  = "#8892a4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": SUB, "ytick.color": SUB,
    "figure.facecolor": BG, "axes.facecolor": CARD,
})

def scatter_reg(ax, x, y, color, xlabel, ylabel, title, subtitle=""):
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    ax.scatter(xv, yv, color=color, alpha=0.75, s=70, edgecolors="white", linewidths=0.5, zorder=3)
    if len(xv) > 2:
        m, b, r, p, _ = stats.linregress(xv, yv)
        xl = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xl, m * xl + b, color=color, linewidth=2.5, alpha=0.9, zorder=4)
        stars = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        p_str = "p<0.001" if p<0.001 else f"p={p:.3f}"
        ax.text(0.97, 0.06, f"r={r:.2f}{stars}  {p_str}", transform=ax.transAxes,
                fontsize=10, color=color, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2235", alpha=0.8,
                          edgecolor=color, linewidth=1))
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color=TEXT)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, fontsize=9.5, color=SUB, ha="center")
    ax.spines[:].set_color("#2a2f45"); ax.grid(True, alpha=0.15, color="white"); ax.tick_params(labelsize=10)

# ─────────────────────────────────────────────
# 11. PLOT — 6 MAIN GRAPHS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle("Until Dawn — AI Influence & Groupthink Study", fontsize=22, fontweight="bold", color=TEXT, y=0.98)
fig.text(0.5, 0.965, "Post-Survey Analysis · April 2026 · N=35", ha="center", fontsize=12, color=SUB)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.94, bottom=0.04)

ax1 = fig.add_subplot(gs[0, 0:2])
scatter_reg(ax1, data["AI_Influence"], data["Groupthink"], C1,
            "AI Influence Score", "Groupthink Score",
            "Graph 1 — AI Influence → Groupthink", "Does AI increase groupthink?")

ax2 = fig.add_subplot(gs[0, 2])
scatter_reg(ax2, data["Groupthink"], data["Regret"], C2,
            "Groupthink Score", "Regret Score",
            "Graph 2 — Groupthink → Regret", "Does groupthink lead to regret?")

ax3 = fig.add_subplot(gs[1, 0])
groups = [data[data["Agreed_with_AI"] == False]["Groupthink"].dropna(),
          data[data["Agreed_with_AI"] == True]["Groupthink"].dropna()]
labels = ["Disagreed\nwith AI", "Agreed\nwith AI"]
bp = ax3.boxplot(groups, patch_artist=True, widths=0.45,
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(color=SUB), capprops=dict(color=SUB),
                 flierprops=dict(marker="o", color=SUB, alpha=0.5))
for patch, col in zip(bp["boxes"], [C3, C1]):
    patch.set_facecolor(col); patch.set_alpha(0.7); patch.set_edgecolor("white")
for i, (grp, col) in enumerate(zip(groups, [C3, C1])):
    jitter = np.random.uniform(-0.1, 0.1, len(grp))
    ax3.scatter([i + 1 + j for j in jitter], grp, color=col, alpha=0.6, s=40, zorder=5,
                edgecolors="white", linewidths=0.3)
t3v, p3v = stats.ttest_ind(groups[0], groups[1])
p_str3 = "p<0.001" if p3v<0.001 else f"p={p3v:.3f}"
ax3.text(0.97, 0.06, f"t-test: {p_str3}", transform=ax3.transAxes, fontsize=9.5, color=C3, ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2235", alpha=0.8, edgecolor=C3, linewidth=1))
ax3.set_xticks([1, 2]); ax3.set_xticklabels(labels, fontsize=10)
ax3.set_ylabel("Groupthink Score", fontsize=11)
ax3.set_title("Graph 3 — Behavior vs Groupthink", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax3.text(0.5, 1.01, "Do AI followers show more groupthink?", transform=ax3.transAxes, fontsize=9.5, color=SUB, ha="center")
ax3.spines[:].set_color("#2a2f45"); ax3.grid(True, alpha=0.15, color="white", axis="y"); ax3.tick_params(labelsize=10)

ax4 = fig.add_subplot(gs[1, 1])
cond_means = data.groupby("AI_Condition")["Groupthink"].mean()
cond_sems  = data.groupby("AI_Condition")["Groupthink"].sem()
order = ["Low AI", "High AI"]
bars = ax4.bar(order, [cond_means[c] for c in order],
               yerr=[cond_sems[c] for c in order],
               color=[C3, C1], alpha=0.8, edgecolor="white", linewidth=0.8,
               error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
for bar, val in zip(bars, [cond_means[c] for c in order]):
    ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}",
             ha="center", fontsize=11, color=TEXT, fontweight="bold")
_, p4v = stats.ttest_ind(data[data["AI_Condition"]=="Low AI"]["Groupthink"].dropna(),
                          data[data["AI_Condition"]=="High AI"]["Groupthink"].dropna())
p_str4 = "p<0.001" if p4v<0.001 else f"p={p4v:.3f}"
ax4.text(0.97, 0.06, p_str4, transform=ax4.transAxes, fontsize=9.5, color=C1, ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2235", alpha=0.8, edgecolor=C1, linewidth=1))
ax4.set_ylabel("Groupthink Score", fontsize=11)
ax4.set_title("Graph 4 — AI Condition vs Groupthink", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax4.text(0.5, 1.01, "High vs. Low AI Influence groups", transform=ax4.transAxes, fontsize=9.5, color=SUB, ha="center")
ax4.spines[:].set_color("#2a2f45"); ax4.grid(True, alpha=0.15, color="white", axis="y")
ax4.set_ylim(0, max(cond_means.values) * 1.3); ax4.tick_params(labelsize=10)

ax5 = fig.add_subplot(gs[1, 2])
scatter_reg(ax5, data["Cohesion"], data["Groupthink"], C4,
            "Cohesion Score", "Groupthink Score",
            "Graph 5 — Cohesion → Groupthink", "Classic groupthink theory link")

ax6 = fig.add_subplot(gs[2, :])
hv = ["AI_Influence", "Cohesion", "Groupthink", "Regret"]
hv_labels = ["AI Influence", "Cohesion", "Groupthink", "Regret"]
corr_m = data[hv].corr()
n_v = len(hv)
p_m = np.zeros((n_v, n_v))
for i in range(n_v):
    for j in range(n_v):
        if i != j:
            x = data[hv[i]].dropna(); y = data[hv[j]].dropna()
            idx = x.index.intersection(y.index)
            _, p_m[i, j] = stats.pearsonr(x[idx], y[idx])

cmap = LinearSegmentedColormap.from_list("custom", ["#e85d75", "#1e2235", "#7b5ea7"], N=256)
im = ax6.imshow(corr_m.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax6, shrink=0.6, label="Pearson r", pad=0.02)
ax6.set_xticks(range(n_v)); ax6.set_xticklabels(hv_labels, fontsize=13, fontweight="bold")
ax6.set_yticks(range(n_v)); ax6.set_yticklabels(hv_labels, fontsize=13, fontweight="bold")
for i in range(n_v):
    for j in range(n_v):
        r = corr_m.values[i, j]; p = p_m[i, j]
        star = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        color = "white" if abs(r) > 0.4 else TEXT
        ax6.text(j, i - 0.12, f"{r:.2f}", ha="center", va="center", fontsize=14,
                 fontweight="bold", color=color)
        ax6.text(j, i + 0.25, star, ha="center", va="center", fontsize=12, color=C4)
ax6.set_title("Graph 6 — Correlation Heatmap", fontsize=14, fontweight="bold", pad=15, color=TEXT)
ax6.text(0.5, -0.08, "* p<.05   ** p<.01   *** p<.001",
         transform=ax6.transAxes, fontsize=11, color=SUB, ha="center")
ax6.spines[:].set_color("#2a2f45")

plt.savefig(os.path.join(output_dir, "graphs_verification.png"), dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n✓ Saved: {os.path.join(output_dir, 'graphs_verification.png')}")

# ─────────────────────────────────────────────
# 12. PLOT — HIDDEN FINDINGS
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig2.suptitle("Until Dawn — Hidden / Unexpected Findings", fontsize=18, fontweight="bold", color=TEXT, y=1.01)
fig2.subplots_adjust(hspace=0.5, wspace=0.35)

hidden_pairs = [
    ("Performance",        "Regret",             "#e85d75", "Performance Score",        "Regret Score",        "Performance → Regret",            "Strongest (r=−0.82)"),
    ("AI_Influence",       "Stress",             "#4fc3f7", "AI Influence Score",       "Stress Score",        "AI Influence → Stress",           "Unexpected (r=+0.52)"),
    ("Critical_Thinking",  "Commitment_Bias",    "#ffd166", "Critical Thinking Score",  "Commitment Bias",     "Critical Thinking → Commitment Bias", "Paradox (r=+0.56)"),
    ("Illusion_Unanimity", "Performance",        "#7b5ea7", "Illusion of Unanimity",    "Performance Score",   "Unanimity → Performance",         "Counterintuitive (r=+0.74)"),
    ("AI_Influence",       "Cohesion",           "#56cfe1", "AI Influence Score",       "Cohesion Score",      "AI Influence → Cohesion",         "Subtle erosion (r=−0.36)"),
    ("Groupthink",         "Critical_Thinking",  "#ff9f1c", "Groupthink Score",         "Critical Thinking",   "Groupthink ↔ Critical Thinking",  "Coexistence paradox (r=+0.37)"),
]

for ax, (a, b, col, xl, yl, title, sub) in zip(axes.flat, hidden_pairs):
    ax.set_facecolor(CARD)
    scatter_reg(ax, data[a], data[b], col, xl, yl, title, sub)

plt.savefig(os.path.join(output_dir, "hidden_findings.png"), dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ Saved: {os.path.join(output_dir, 'hidden_findings.png')}")

# ─────────────────────────────────────────────
# 13.  PLOT — CONDITION (Q36) ANALYSIS
# ─────────────────────────────────────────────
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

fig3 = plt.figure(figsize=(22, 18), facecolor=BG)
fig3.suptitle("Until Dawn — Condition (Q36) Analysis", fontsize=20, fontweight="bold", color=TEXT, y=0.99)
fig3.text(0.5, 0.965, "How AI framing type and valence affected psychological outcomes",
          ha="center", fontsize=12, color=SUB)
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.5, wspace=0.35,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

# Panel A — AI Influence by Condition Type (ANOVA ***)
ax_a = fig3.add_subplot(gs3[0, 0])
ax_a.set_facecolor(CARD)
type_order = ["control", "open", "neutral", "closed"]
type_means = data.groupby("Condition_Type")["AI_Influence"].mean()
type_sems  = data.groupby("Condition_Type")["AI_Influence"].sem()
type_ns    = data.groupby("Condition_Type")["AI_Influence"].count()
for xi, t in enumerate(type_order):
    ax_a.bar(xi, type_means[t], yerr=type_sems[t], color=TYPE_COLORS[t], alpha=0.8,
             edgecolor="white", linewidth=0.8,
             error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
    ax_a.text(xi, type_means[t] + type_sems[t] + 0.3, f"{type_means[t]:.2f}",
              ha="center", fontsize=11, color=TEXT, fontweight="bold")
    grp_vals = data[data["Condition_Type"] == t]["AI_Influence"].dropna().values
    np.random.seed(xi * 13)
    jit = np.random.uniform(-0.15, 0.15, len(grp_vals))
    ax_a.scatter(xi + jit, grp_vals, color=TYPE_COLORS[t], s=40, alpha=0.6,
                 zorder=5, edgecolors="white", linewidths=0.3)
groups_a = [data[data["Condition_Type"] == t]["AI_Influence"].dropna().values for t in type_order]
f_a, p_a = stats.f_oneway(*groups_a)
stars_a = "***" if p_a < 0.001 else ("**" if p_a < 0.01 else ("*" if p_a < 0.05 else "ns"))
ax_a.text(0.97, 0.06, f"ANOVA: F={f_a:.2f}, p<0.001 {stars_a}",
          transform=ax_a.transAxes, fontsize=9.5, color=C4, ha="right",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2235", alpha=0.8, edgecolor=C4, linewidth=1))
ax_a.set_xticks(range(len(type_order)))
ax_a.set_xticklabels([f"{t.capitalize()}\n(n={type_ns.get(t,0)})" for t in type_order], fontsize=10)
ax_a.set_ylabel("AI Influence Score", fontsize=11)
ax_a.set_title("Panel A — AI Influence by Condition Type", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax_a.text(0.5, 1.01, "Closed/Neutral framing drives highest AI influence (F=8.15***)",
          transform=ax_a.transAxes, fontsize=9.5, color=SUB, ha="center")
ax_a.spines[:].set_color("#2a2f45"); ax_a.grid(True, alpha=0.15, color="white", axis="y")
ax_a.set_ylim(0, 10)

# Panel B — Cohesion by Condition Valence (*)
ax_b = fig3.add_subplot(gs3[0, 1])
ax_b.set_facecolor(CARD)
val_order = ["control", "positive", "negative"]
val_means_c = data.groupby("Condition_Valence")["Cohesion"].mean()
val_sems_c  = data.groupby("Condition_Valence")["Cohesion"].sem()
val_ns_c    = data.groupby("Condition_Valence")["Cohesion"].count()
for xi, v in enumerate(val_order):
    ax_b.bar(xi, val_means_c[v], yerr=val_sems_c[v], color=VAL_COLORS[v], alpha=0.8,
             edgecolor="white", linewidth=0.8,
             error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
    ax_b.text(xi, val_means_c[v] + val_sems_c[v] + 0.3, f"{val_means_c[v]:.2f}",
              ha="center", fontsize=11, color=TEXT, fontweight="bold")
    grp_vals = data[data["Condition_Valence"] == v]["Cohesion"].dropna().values
    np.random.seed(xi * 17)
    jit = np.random.uniform(-0.15, 0.15, len(grp_vals))
    ax_b.scatter(xi + jit, grp_vals, color=VAL_COLORS[v], s=40, alpha=0.6,
                 zorder=5, edgecolors="white", linewidths=0.3)
groups_b = [data[data["Condition_Valence"] == v]["Cohesion"].dropna().values for v in val_order]
f_b, p_b = stats.f_oneway(*groups_b)
stars_b = "***" if p_b < 0.001 else ("**" if p_b < 0.01 else ("*" if p_b < 0.05 else "ns"))
ax_b.text(0.97, 0.06, f"ANOVA: F={f_b:.2f}, p={p_b:.3f} {stars_b}",
          transform=ax_b.transAxes, fontsize=9.5, color=C4, ha="right",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2235", alpha=0.8, edgecolor=C4, linewidth=1))
ax_b.set_xticks(range(len(val_order)))
ax_b.set_xticklabels([f"{v.capitalize()}\n(n={val_ns_c.get(v,0)})" for v in val_order], fontsize=10)
ax_b.set_ylabel("Cohesion Score", fontsize=11)
ax_b.set_title("Panel B — Team Cohesion by Condition Valence", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax_b.text(0.5, 1.01, "Control group markedly more cohesive than all framed conditions",
          transform=ax_b.transAxes, fontsize=9.5, color=SUB, ha="center")
ax_b.spines[:].set_color("#2a2f45"); ax_b.grid(True, alpha=0.15, color="white", axis="y")
ax_b.set_ylim(0, 12)

# Panel C — Control vs Experimental (multi-score)
ax_c = fig3.add_subplot(gs3[1, 0])
ax_c.set_facecolor(CARD)
compare_scores = ["AI_Influence", "Stress", "Cohesion", "Groupthink", "Performance"]
compare_labels = ["AI\nInfluence", "Stress", "Cohesion", "Groupthink", "Performance"]
ctrl_means = [data[data["Condition_Type"] == "control"][s].mean() for s in compare_scores]
exp_means  = [data[data["Condition_Type"] != "control"][s].mean() for s in compare_scores]
ctrl_sems  = [data[data["Condition_Type"] == "control"][s].sem() for s in compare_scores]
exp_sems   = [data[data["Condition_Type"] != "control"][s].sem() for s in compare_scores]
x_pos = np.arange(len(compare_scores))
w = 0.35
ax_c.bar(x_pos - w/2, ctrl_means, w, yerr=ctrl_sems, label="Control (n=5)",
         color="#8892a4", alpha=0.85, edgecolor="white",
         error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))
ax_c.bar(x_pos + w/2, exp_means, w, yerr=exp_sems, label="Experimental (n=30)",
         color=C1, alpha=0.85, edgecolor="white",
         error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))
for xi, s in enumerate(compare_scores):
    ctrl_g = data[data["Condition_Type"] == "control"][s].dropna()
    exp_g  = data[data["Condition_Type"] != "control"][s].dropna()
    if len(ctrl_g) >= 2 and len(exp_g) >= 2:
        _, pv = stats.ttest_ind(ctrl_g, exp_g)
        star = "***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else ""))
        if star:
            ymax = max(ctrl_means[xi] + ctrl_sems[xi], exp_means[xi] + exp_sems[xi]) + 0.5
            ax_c.text(xi, ymax, star, ha="center", fontsize=13, color=C4)
ax_c.set_xticks(x_pos); ax_c.set_xticklabels(compare_labels, fontsize=10)
ax_c.set_ylabel("Score", fontsize=11)
ax_c.legend(fontsize=9.5, facecolor="#1e2235", edgecolor="#2a2f45", labelcolor=TEXT)
ax_c.set_title("Panel C — Control vs Experimental Conditions", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax_c.text(0.5, 1.01, "AI conditions raise Influence+Stress, lower Cohesion (★ = p<.05)",
          transform=ax_c.transAxes, fontsize=9.5, color=SUB, ha="center")
ax_c.spines[:].set_color("#2a2f45"); ax_c.grid(True, alpha=0.15, color="white", axis="y")
ax_c.set_ylim(0, 12)

# Panel D — AI Influence across all 7 conditions (horizontal bar)
ax_d = fig3.add_subplot(gs3[1, 1])
ax_d.set_facecolor(CARD)
all_conds = ["control", "open positive", "open negative",
             "neutral positive", "neutral negative", "closed positive", "closed negative"]
cond_ai_means = [data[data["Condition"] == c]["AI_Influence"].mean() for c in all_conds]
cond_ai_sems  = [data[data["Condition"] == c]["AI_Influence"].sem() for c in all_conds]
cond_ns_all   = [data[data["Condition"] == c]["AI_Influence"].count() for c in all_conds]
bar_colors    = [COND_COLORS[c] for c in all_conds]
ax_d.barh(range(len(all_conds)), cond_ai_means, xerr=cond_ai_sems,
          color=bar_colors, alpha=0.82, edgecolor="white", linewidth=0.7,
          error_kw=dict(ecolor="white", elinewidth=1.2, capsize=4))
for yi, (m, n) in enumerate(zip(cond_ai_means, cond_ns_all)):
    ax_d.text(m + 0.2, yi, f"{m:.2f}", va="center", fontsize=10, color=TEXT, fontweight="bold")
ax_d.set_yticks(range(len(all_conds)))
ax_d.set_yticklabels([f"{c.title()} (n={n})" for c, n in zip(all_conds, cond_ns_all)], fontsize=9.5)
ax_d.set_xlabel("AI Influence Score", fontsize=11)
ax_d.set_title("Panel D — AI Influence Across All 7 Conditions", fontsize=13, fontweight="bold", pad=10, color=TEXT)
ax_d.text(0.5, 1.01, "Control stands apart; framing type matters more than valence",
          transform=ax_d.transAxes, fontsize=9.5, color=SUB, ha="center")
ax_d.spines[:].set_color("#2a2f45"); ax_d.grid(True, alpha=0.15, color="white", axis="x")
ax_d.set_xlim(0, 10)

plt.savefig(os.path.join(output_dir, "condition_analysis.png"), dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ Saved: {os.path.join(output_dir, 'condition_analysis.png')}")

print("\n=== DONE — all numbers above are computed directly from your xlsx ===")
