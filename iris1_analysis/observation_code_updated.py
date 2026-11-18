import os
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ============================================================
# DATA LOADING & SCORE COMPUTATION
# ============================================================
df = pd.read_excel('Observation Sheet_April 6, 2026_12.19.xlsx', header=0)
data = df.iloc[1:].copy().reset_index(drop=True)

# Column mapping (new column Q3=Framing, Q4=Team)
# Q3 = Framing Condition (open/closed/neutral × positive/negative + Control)
# Q4 = Team ID (was Q3 in original)

output_dir = 'obs_survey_figs'
os.makedirs(output_dir, exist_ok=True)

def clean_yn(val):
    if pd.isna(val): return 0
    return 1 if str(val).strip().lower() in ['yes','yea','y'] else 0

def has_ai_rec(val):
    if pd.isna(val): return False
    return str(val).strip().lower() not in ['no','none','none ','']

def parse_alive(val):
    if pd.isna(val): return np.nan
    v = str(val).lower().strip()
    if 'fate unclear' in v: return 1
    m = re.match(r'^(\d+)', v)
    if m: return int(m.group(1))
    names = ['ashley','matt','chris','emily']
    c = 0
    for n in names:
        idx = v.find(n)
        if idx == -1: continue
        if 'died' not in v[idx:idx+20] and 'dead' not in v[idx:idx+20]:
            c += 1
    return c if c > 0 else np.nan

def parse_framing(val):
    """Parse Q3 into openness and valence dimensions."""
    if pd.isna(val): return np.nan, np.nan
    v = str(val).strip().lower()
    if v == 'control': return 'Control', 'Control'
    openness = ('open' if 'open' in v else
                'closed' if 'closed' in v else
                'neutral' if 'neutral' in v else np.nan)
    valence  = ('positive' if 'positive' in v else
                'negative' if 'negative' in v else np.nan)
    return openness, valence

results = []
for idx, row in data.iterrows():
    team      = row['Q4']          # Team ID (shifted from Q3)
    framing   = str(row['Q3']).strip() if not pd.isna(row['Q3']) else np.nan
    if pd.isna(team): continue

    openness, valence = parse_framing(framing)

    agreements, dissents_exp, dissents_sup, alternatives, sources, ai_recs = [],[],[],[],[],[]
    for d in range(1, 14):
        p = f'Q5_{d}_'
        ai_recs.append(has_ai_rec(row.get(p+'2')))
        agreements.append(clean_yn(row.get(p+'3')) if not pd.isna(row.get(p+'3')) else 0)
        dissents_exp.append(clean_yn(row.get(p+'5')) if not pd.isna(row.get(p+'5')) else 0)
        dissents_sup.append(clean_yn(row.get(p+'6')) if not pd.isna(row.get(p+'6')) else 0)
        alternatives.append(clean_yn(row.get(p+'7')) if not pd.isna(row.get(p+'7')) else 0)
        src = str(row.get(p+'8', '')).strip().lower()
        sources.append('AI' if ('ai' in src and 'human' not in src)
                        else ('Human' if 'human' in src
                        else ('Mixed' if 'mixed' in src else None)))

    n_ai = sum(ai_recs)
    ai_influence  = sum(agreements) / n_ai if n_ai > 0 else 0
    groupthink    = max(0, min(1, (sum(agreements) + sum(dissents_sup) - sum(alternatives)) / 13))
    cohesion      = 1 - (sum(dissents_exp) / 13)
    alive         = parse_alive(row.get('Q7'))
    regret        = (4 - alive) / 4 if (alive is not None and not np.isnan(alive)) else np.nan
    ai_src        = sum(1 for s in sources if s == 'AI')
    hu_src        = sum(1 for s in sources if s == 'Human')
    behavior      = ('Agreed' if ai_src > hu_src else
                     'Disagreed' if hu_src > ai_src else 'Mixed')
    condition     = 'AI Present' if n_ai > 0 else 'No AI'
    consec        = 0
    for a in agreements:
        if a == 1: consec += 1
        else: break

    # Framing valence score: positive=+1, negative=-1, control=0
    valence_score = (1 if valence == 'positive' else
                    -1 if valence == 'negative' else 0)
    # Openness score: open=1, neutral=0, closed=-1
    openness_score = (1 if openness == 'open' else
                     -1 if openness == 'closed' else 0)

    results.append({
        'Team'            : str(team).strip(),
        'Framing'         : framing,
        'Openness'        : openness,
        'Valence'         : valence,
        'Valence_Score'   : valence_score,
        'Openness_Score'  : openness_score,
        'AI_Influence'    : ai_influence,
        'Groupthink'      : groupthink,
        'Cohesion'        : cohesion,
        'Regret'          : regret,
        'Behavior'        : behavior,
        'Condition'       : condition,
        'N_AI_Recs'       : n_ai,
        'N_Agreements'    : sum(agreements),
        'N_Dissent_Exp'   : sum(dissents_exp),
        'N_Dissent_Sup'   : sum(dissents_sup),
        'N_Alternatives'  : sum(alternatives),
        'Consec_Agree'    : consec,
    })

df_raw = pd.DataFrame(results)

# Average the two observers per team
dt = df_raw.groupby('Team').agg({
    'AI_Influence'   : 'mean',
    'Groupthink'     : 'mean',
    'Cohesion'       : 'mean',
    'Regret'         : 'mean',
    'N_AI_Recs'      : 'mean',
    'N_Agreements'   : 'mean',
    'N_Dissent_Exp'  : 'mean',
    'N_Dissent_Sup'  : 'mean',
    'N_Alternatives' : 'mean',
    'Consec_Agree'   : 'mean',
    'Valence_Score'  : 'mean',
    'Openness_Score' : 'mean',
    'Behavior'       : lambda x: x.mode()[0],
    'Condition'      : lambda x: x.mode()[0],
    'Framing'        : lambda x: x.mode()[0],
    'Openness'       : lambda x: x.mode()[0],
    'Valence'        : lambda x: x.mode()[0],
}).reset_index()

# Normalize framing label
dt['Framing'] = dt['Framing'].str.strip().str.lower()
dt['Framing'] = dt['Framing'].replace({'control': 'Control',
                                        'closed negative': 'Closed Negative',
                                        'closed positive': 'Closed Positive',
                                        'open negative':   'Open Negative',
                                        'open positive':   'Open Positive',
                                        'neutral negative':'Neutral Negative',
                                        'neutral positive':'Neutral Positive'})

print("=" * 60)
print(f"TEAMS IN ANALYSIS: {len(dt)}")
print("=" * 60)
print("\nRAW TEAM-LEVEL SCORES:")
print(dt[['Team','Framing','AI_Influence','Groupthink','Cohesion',
          'Regret','Behavior','Condition','N_AI_Recs']].to_string(index=False))

# ============================================================
# SIGNIFICANCE TESTS
# ============================================================
def pearson(label, x_series, y_series):
    df_pair = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()
    n = len(df_pair)
    r, p = stats.pearsonr(df_pair['x'], df_pair['y'])
    sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    print(f"  {label:<55} r={r:+.3f}  p={p:.4f}  n={n}  {sig}")
    return r, p, n

print("\n" + "=" * 60)
print("SIGNIFICANCE TESTS")
print("=" * 60)

print("\n--- Original 6 Graphs ---")
pearson("G1: AI_Influence vs Groupthink",    dt['AI_Influence'], dt['Groupthink'])
pearson("G2: Groupthink vs Regret",          dt['Groupthink'],   dt['Regret'])
pearson("G5: Cohesion vs Groupthink",        dt['Cohesion'],     dt['Groupthink'])

print("\n  G3: Groupthink by Behavior (ANOVA / group means)")
for b in ['Agreed','Mixed','Disagreed']:
    sub = dt[dt['Behavior']==b]['Groupthink']
    print(f"    {b:<12}: mean={sub.mean():.3f}  sd={sub.std():.3f}  n={len(sub)}")
ag = dt[dt['Behavior']=='Agreed']['Groupthink'].values
dg = dt[dt['Behavior']=='Disagreed']['Groupthink'].values
t, p = stats.ttest_ind(ag, dg)
print(f"    t-test Agreed vs Disagreed: t={t:.3f}  p={p:.4f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

print("\n  G4: AI Present vs No AI (t-test on Groupthink)")
ai_g   = dt[dt['Condition']=='AI Present']['Groupthink'].values
noai_g = dt[dt['Condition']=='No AI']['Groupthink'].values
t, p = stats.ttest_ind(ai_g, noai_g)
print(f"    AI Present: mean={ai_g.mean():.3f}  n={len(ai_g)}")
print(f"    No AI:      mean={noai_g.mean():.3f}  n={len(noai_g)}")
print(f"    t={t:.3f}  p={p:.4f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

print("\n--- NEW: Framing Condition Analysis ---")
print("\n  Framing group means (Groupthink, AI_Influence, Regret):")
for f in dt['Framing'].dropna().unique():
    sub = dt[dt['Framing']==f]
    gt  = sub['Groupthink'].mean()
    ai  = sub['AI_Influence'].mean()
    rg  = sub['Regret'].dropna().mean()
    print(f"    {f:<22}: GT={gt:.3f}  AI_Inf={ai:.3f}  Regret={rg:.3f}  n={len(sub)}")

# One-way ANOVA: Framing vs Groupthink
framing_groups_gt = [dt[dt['Framing']==f]['Groupthink'].dropna().values
                     for f in dt['Framing'].dropna().unique() if len(dt[dt['Framing']==f]) >= 2]
F_gt, p_gt = stats.f_oneway(*framing_groups_gt)
print(f"\n  ANOVA Framing → Groupthink:    F={F_gt:.3f}  p={p_gt:.4f}  {'***' if p_gt<0.001 else '**' if p_gt<0.01 else '*' if p_gt<0.05 else 'ns'}")

framing_groups_ai = [dt[dt['Framing']==f]['AI_Influence'].dropna().values
                     for f in dt['Framing'].dropna().unique() if len(dt[dt['Framing']==f]) >= 2]
F_ai, p_ai = stats.f_oneway(*framing_groups_ai)
print(f"  ANOVA Framing → AI_Influence:  F={F_ai:.3f}  p={p_ai:.4f}  {'***' if p_ai<0.001 else '**' if p_ai<0.01 else '*' if p_ai<0.05 else 'ns'}")

framing_groups_rg = [dt[dt['Framing']==f]['Regret'].dropna().values
                     for f in dt['Framing'].dropna().unique() if len(dt[dt['Framing']==f]['Regret'].dropna()) >= 2]
F_rg, p_rg = stats.f_oneway(*framing_groups_rg)
print(f"  ANOVA Framing → Regret:        F={F_rg:.3f}  p={p_rg:.4f}  {'***' if p_rg<0.001 else '**' if p_rg<0.01 else '*' if p_rg<0.05 else 'ns'}")

print("\n  Valence (positive vs negative) — t-tests:")
pos = dt[dt['Valence']=='positive']
neg = dt[dt['Valence']=='negative']
for metric in ['Groupthink','AI_Influence','Regret','Cohesion']:
    pv = pos[metric].dropna().values
    nv = neg[metric].dropna().values
    if len(pv) >= 2 and len(nv) >= 2:
        t, p = stats.ttest_ind(pv, nv)
        print(f"    {metric:<15}: pos={pv.mean():.3f}  neg={nv.mean():.3f}  t={t:.3f}  p={p:.4f}  {'*' if p<0.05 else 'ns'}")

print("\n  Openness (open vs closed) — t-tests:")
opn = dt[dt['Openness']=='open']
cls = dt[dt['Openness']=='closed']
for metric in ['Groupthink','AI_Influence','Regret','Cohesion']:
    ov = opn[metric].dropna().values
    cv = cls[metric].dropna().values
    if len(ov) >= 2 and len(cv) >= 2:
        t, p = stats.ttest_ind(ov, cv)
        print(f"    {metric:<15}: open={ov.mean():.3f}  closed={cv.mean():.3f}  t={t:.3f}  p={p:.4f}  {'*' if p<0.05 else 'ns'}")

print("\n  Continuous framing correlations:")
pearson("Valence_Score  vs Groupthink",   dt['Valence_Score'],  dt['Groupthink'])
pearson("Valence_Score  vs AI_Influence", dt['Valence_Score'],  dt['AI_Influence'])
pearson("Valence_Score  vs Regret",       dt['Valence_Score'],  dt['Regret'])
pearson("Openness_Score vs Groupthink",   dt['Openness_Score'], dt['Groupthink'])
pearson("Openness_Score vs AI_Influence", dt['Openness_Score'], dt['AI_Influence'])
pearson("Openness_Score vs Regret",       dt['Openness_Score'], dt['Regret'])
pearson("Valence_Score  vs Cohesion",     dt['Valence_Score'],  dt['Cohesion'])
pearson("Openness_Score vs Cohesion",     dt['Openness_Score'], dt['Cohesion'])

print("\n--- Hidden Findings (original) ---")
pearson("F2: Consec_Agree vs Groupthink",    dt['Consec_Agree'],    dt['Groupthink'])
pearson("F3: N_Alternatives vs Groupthink",  dt['N_Alternatives'],  dt['Groupthink'])
pearson("F4: N_AI_Recs vs Groupthink",       dt['N_AI_Recs'],       dt['Groupthink'])
pearson("F6a: N_Dissent_Exp vs Groupthink",  dt['N_Dissent_Exp'],   dt['Groupthink'])
pearson("F6b: N_Dissent_Sup vs Groupthink",  dt['N_Dissent_Sup'],   dt['Groupthink'])

# ============================================================
# PLOTS — Original 6 + 4 new framing plots (10 total)
# ============================================================
BG   = '#0F0F1A'; CARD = '#161628'; TEXT = '#E8E8F0'; SUB  = '#9090B0'
GRID = '#2A2A3F'; C1   = '#7C5CFC'; C2   = '#FC5C7D'; C3   = '#43E97B'
C4   = '#F7971E'; C5   = '#00C9FF'; C6   = '#FFD700'; C7   = '#FF6B9D'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'text.color': TEXT,
    'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'axes.edgecolor': GRID, 'figure.facecolor': BG, 'axes.facecolor': CARD,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig = plt.figure(figsize=(22, 40), facecolor=BG)
fig.suptitle('AI Influence, Groupthink & Framing Condition — Full Dashboard', fontsize=24,
             fontweight='bold', color=TEXT, y=0.995)
fig.text(0.5, 0.988, f'n = {len(dt)} teams  |  Includes new Framing Condition analysis',
         fontsize=12, color=SUB, ha='center')

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.08, right=0.95, top=0.982, bottom=0.02)

def style_ax(ax, title):
    ax.set_facecolor(CARD); ax.grid(True, alpha=0.4, linewidth=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT, pad=10)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

def add_reg(ax, x, y, color):
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < 3: return
    slope, intercept, r, p, _ = stats.linregress(xc, yc)
    xline = np.linspace(xc.min(), xc.max(), 100)
    ax.plot(xline, slope*xline + intercept, '--', color=color, lw=2, alpha=0.85)
    ps = 'p<0.001' if p<0.001 else (f'p={p:.3f}' if p<0.05 else f'p={p:.2f}')
    ax.text(0.97, 0.06, f'r = {r:.2f}  {ps}', transform=ax.transAxes,
            fontsize=11, color=color, ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, edgecolor=color, alpha=0.85))

# ── G1: AI Influence → Groupthink ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Graph 1: AI Influence → Groupthink')
sc = ax1.scatter(dt['AI_Influence'], dt['Groupthink'], c=dt['Groupthink'],
                 cmap='plasma', s=100, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_reg(ax1, dt['AI_Influence'].values, dt['Groupthink'].values, C2)
ax1.set_xlabel('AI Influence Score'); ax1.set_ylabel('Groupthink Score')
plt.colorbar(sc, ax=ax1).outline.set_edgecolor(GRID)

# ── G2: Groupthink → Regret ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Graph 2: Groupthink → Regret')
dt_r = dt.dropna(subset=['Regret'])
sc2 = ax2.scatter(dt_r['Groupthink'], dt_r['Regret'], c=dt_r['Regret'],
                  cmap='RdYlGn_r', s=100, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_reg(ax2, dt_r['Groupthink'].values, dt_r['Regret'].values, C4)
ax2.set_xlabel('Groupthink Score'); ax2.set_ylabel('Regret (deaths/4)')
plt.colorbar(sc2, ax=ax2).outline.set_edgecolor(GRID)

# ── G3: Behavior vs Groupthink ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'Graph 3: Behavior vs Groupthink')
beh_order  = ['Agreed','Mixed','Disagreed']
beh_colors = [C1, C4, C5]
beh_data   = [dt[dt['Behavior']==b]['Groupthink'].values for b in beh_order]
beh_labels = [f'{b}\n(n={len(dt[dt["Behavior"]==b])})' for b in beh_order]
bps = ax3.boxplot(beh_data, patch_artist=True, widths=0.45,
                  medianprops=dict(color='white', linewidth=2.5),
                  whiskerprops=dict(color=SUB, linewidth=1.5),
                  capprops=dict(color=SUB, linewidth=1.5),
                  flierprops=dict(marker='o', color=SUB, alpha=0.5, markersize=5))
for patch, color in zip(bps['boxes'], beh_colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for i, (bd, c) in enumerate(zip(beh_data, beh_colors)):
    jitter = np.random.normal(0, 0.07, size=len(bd))
    ax3.scatter(np.full(len(bd), i+1)+jitter, bd, color=c, alpha=0.7,
                s=45, zorder=5, edgecolors='white', linewidth=0.4)
ax3.set_xticks([1,2,3]); ax3.set_xticklabels(beh_labels, fontsize=10)
ax3.set_ylabel('Groupthink Score')
t_val, p_val = stats.ttest_ind(beh_data[0], beh_data[2])
ps = 'p<0.001' if p_val<0.001 else f'p={p_val:.3f}'
ax3.text(0.97, 0.06, f'Agreed vs Disagreed: {ps}', transform=ax3.transAxes,
         fontsize=10, color=C2, ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=C2, alpha=0.85))

# ── G4: AI Condition vs Groupthink ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'Graph 4: AI Condition vs Groupthink')
cond_order  = ['AI Present','No AI']
cond_colors = [C1, C3]
cond_means  = [dt[dt['Condition']==c]['Groupthink'].mean() for c in cond_order]
cond_sems   = [dt[dt['Condition']==c]['Groupthink'].sem()  for c in cond_order]
cond_ns     = [len(dt[dt['Condition']==c]) for c in cond_order]
ax4.bar([f'{c}\n(n={n})' for c,n in zip(cond_order,cond_ns)], cond_means,
        color=cond_colors, alpha=0.8, width=0.45, edgecolor='white', linewidth=0.7)
ax4.errorbar([0,1], cond_means, yerr=cond_sems, fmt='none',
             color='white', capsize=6, linewidth=2, capthick=2)
for i, (c, col) in enumerate(zip(cond_order, cond_colors)):
    vals = dt[dt['Condition']==c]['Groupthink'].values
    jitter = np.random.normal(0, 0.06, size=len(vals))
    ax4.scatter(np.full(len(vals),i)+jitter, vals, color=col,
                alpha=0.6, s=50, zorder=5, edgecolors='white', linewidth=0.4)
ax4.set_ylabel('Groupthink Score')
t_val, p_val = stats.ttest_ind(ai_g, noai_g)
ps = 'p<0.001' if p_val<0.001 else (f'p={p_val:.3f}' if p_val<0.05 else f'p={p_val:.2f}')
ax4.text(0.5, 0.92, f't-test: {ps}', transform=ax4.transAxes, fontsize=11,
         color=C4, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=C4, alpha=0.85))

# ── G5: Cohesion → Groupthink ──────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, 'Graph 5: Cohesion → Groupthink')
sc5 = ax5.scatter(dt['Cohesion'], dt['Groupthink'], c=dt['AI_Influence'],
                  cmap='cool', s=100, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_reg(ax5, dt['Cohesion'].values, dt['Groupthink'].values, C3)
ax5.set_xlabel('Cohesion Score'); ax5.set_ylabel('Groupthink Score')
cb5 = plt.colorbar(sc5, ax=ax5); cb5.set_label('AI Influence', color=SUB, fontsize=10)
cb5.outline.set_edgecolor(GRID)

# ── G6: Correlation Heatmap (extended with Framing) ────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Graph 6: Correlation Heatmap (+ Framing)')
corr_cols   = ['AI_Influence','Cohesion','Groupthink','Regret','Valence_Score','Openness_Score']
corr_labels = ['AI Influence','Cohesion','Groupthink','Regret','Valence','Openness']
corr_matrix = dt[corr_cols].corr()
im = ax6.imshow(corr_matrix.values, cmap='Oranges', vmin=-1, vmax=1, aspect='auto')
ax6.set_xticks(range(len(corr_labels))); ax6.set_yticks(range(len(corr_labels)))
ax6.set_xticklabels(corr_labels, rotation=35, ha='right', fontsize=10)
ax6.set_yticklabels(corr_labels, fontsize=10)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        v = corr_matrix.values[i,j]
        ax6.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11,
                 fontweight='bold', color='white' if abs(v) > 0.5 else TEXT)
plt.colorbar(im, ax=ax6).outline.set_edgecolor(GRID)

# ── NEW G7: Framing Condition vs Groupthink (grouped bar) ──
ax7 = fig.add_subplot(gs[3, 0])
style_ax(ax7, 'Graph 7 (NEW): Framing Condition vs Groupthink')
framing_order = ['Control','Open Positive','Open Negative',
                 'Neutral Positive','Neutral Negative',
                 'Closed Positive','Closed Negative']
framing_colors = [C5, C3, C2, C6, C4, C1, C7]
f_means = []; f_sems = []; f_labels = []; f_cols = []
for f, c in zip(framing_order, framing_colors):
    sub = dt[dt['Framing']==f]['Groupthink']
    if len(sub) > 0:
        f_means.append(sub.mean()); f_sems.append(sub.sem())
        f_labels.append(f'{f}\n(n={len(sub)})'); f_cols.append(c)
x_pos = range(len(f_means))
bars = ax7.bar(x_pos, f_means, color=f_cols, alpha=0.82,
               width=0.55, edgecolor='white', linewidth=0.7)
ax7.errorbar(list(x_pos), f_means, yerr=f_sems, fmt='none',
             color='white', capsize=5, linewidth=1.8, capthick=1.8)
ax7.set_xticks(list(x_pos)); ax7.set_xticklabels(f_labels, fontsize=8)
ax7.set_ylabel('Mean Groupthink Score')
ax7.text(0.97, 0.93, f'ANOVA p={p_gt:.3f}', transform=ax7.transAxes,
         fontsize=10, color=C4, ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=C4, alpha=0.85))

# ── NEW G8: Framing Condition vs AI Influence ──────────────
ax8 = fig.add_subplot(gs[3, 1])
style_ax(ax8, 'Graph 8 (NEW): Framing Condition vs AI Influence')
fi_means = []; fi_sems = []; fi_labels2 = []; fi_cols = []
for f, c in zip(framing_order, framing_colors):
    sub = dt[dt['Framing']==f]['AI_Influence']
    if len(sub) > 0:
        fi_means.append(sub.mean()); fi_sems.append(sub.sem())
        fi_labels2.append(f'{f}\n(n={len(sub)})'); fi_cols.append(c)
x_pos2 = range(len(fi_means))
ax8.bar(x_pos2, fi_means, color=fi_cols, alpha=0.82,
        width=0.55, edgecolor='white', linewidth=0.7)
ax8.errorbar(list(x_pos2), fi_means, yerr=fi_sems, fmt='none',
             color='white', capsize=5, linewidth=1.8, capthick=1.8)
ax8.set_xticks(list(x_pos2)); ax8.set_xticklabels(fi_labels2, fontsize=8)
ax8.set_ylabel('Mean AI Influence Score')
ax8.text(0.97, 0.93, f'ANOVA p={p_ai:.3f}', transform=ax8.transAxes,
         fontsize=10, color=C4, ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=C4, alpha=0.85))

# ── NEW G9: Valence (pos vs neg) — Groupthink & AI side-by-side ─
ax9 = fig.add_subplot(gs[4, 0])
style_ax(ax9, 'Graph 9 (NEW): Valence × Openness → Groupthink')
dt_f = dt[dt['Valence'].isin(['positive','negative']) &
          dt['Openness'].isin(['open','neutral','closed'])]
openness_cats = ['open','neutral','closed']
valence_cats  = ['positive','negative']
oc_colors     = {('open','positive'): C3, ('open','negative'): C2,
                 ('neutral','positive'): C6, ('neutral','negative'): C4,
                 ('closed','positive'): C1, ('closed','negative'): C7}
x_tick_labels = []; x_tick_pos = []; bar_idx = 0
for oi, op in enumerate(openness_cats):
    for vi, va in enumerate(valence_cats):
        sub = dt_f[(dt_f['Openness']==op) & (dt_f['Valence']==va)]['Groupthink']
        if len(sub) == 0: bar_idx += 1; continue
        col = oc_colors.get((op,va), C5)
        ax9.bar(bar_idx, sub.mean(), color=col, alpha=0.82, width=0.7,
                edgecolor='white', linewidth=0.7)
        if len(sub) > 1:
            ax9.errorbar(bar_idx, sub.mean(), yerr=sub.sem(), fmt='none',
                         color='white', capsize=4, linewidth=1.5)
        lbl = f'{op[:3]}\n{va[:3]}\n(n={len(sub)})'
        x_tick_labels.append(lbl); x_tick_pos.append(bar_idx)
        # scatter jitter
        jitter = np.random.normal(0, 0.08, size=len(sub))
        ax9.scatter(np.full(len(sub), bar_idx)+jitter, sub.values,
                    color=col, alpha=0.7, s=40, zorder=5,
                    edgecolors='white', linewidth=0.4)
        bar_idx += 1
    bar_idx += 0.5  # gap between openness groups
ax9.set_xticks(x_tick_pos); ax9.set_xticklabels(x_tick_labels, fontsize=9)
ax9.set_ylabel('Groupthink Score')

# ── NEW G10: Framing vs Regret ─────────────────────────────
ax10 = fig.add_subplot(gs[4, 1])
style_ax(ax10, 'Graph 10 (NEW): Framing Condition vs Regret')
fr_means = []; fr_sems = []; fr_labels = []; fr_cols = []
for f, c in zip(framing_order, framing_colors):
    sub = dt[dt['Framing']==f]['Regret'].dropna()
    if len(sub) > 0:
        fr_means.append(sub.mean()); fr_sems.append(sub.sem() if len(sub)>1 else 0)
        fr_labels.append(f'{f}\n(n={len(sub)})'); fr_cols.append(c)
x_pos3 = range(len(fr_means))
ax10.bar(x_pos3, fr_means, color=fr_cols, alpha=0.82,
         width=0.55, edgecolor='white', linewidth=0.7)
ax10.errorbar(list(x_pos3), fr_means, yerr=fr_sems, fmt='none',
              color='white', capsize=5, linewidth=1.8, capthick=1.8)
ax10.set_xticks(list(x_pos3)); ax10.set_xticklabels(fr_labels, fontsize=8)
ax10.set_ylabel('Mean Regret (deaths/4)')
ax10.text(0.97, 0.93, f'ANOVA p={p_rg:.3f}', transform=ax10.transAxes,
          fontsize=10, color=C4, ha='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=C4, alpha=0.85))

dashboard_path = os.path.join(output_dir, 'verification_dashboard_updated.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
# print(f"\nDashboard saved to {dashboard_path}")
print("Done..")
