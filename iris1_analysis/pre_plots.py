import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR  = "pre_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel('Until_Dawn_Pre-survey_April_6__2026_12_17__1_.xlsx', header=1)
df.columns = ['TeamID','Condition','Age','Gender','Race','PlayedGames','GameFrequency','AIFamiliarity','PlayedUntilDawn','UntilDawnFamiliarity']

# Normalize condition column to Title Case (handles 'Control'/'control' inconsistency)
df['Condition'] = df['Condition'].str.strip().str.title()

freq_order = {'Rarely (a few times a year)':1,'Occasionally (monthly)':2,'Regularly (weekly or more)':3}
ai_order   = {'Slightly familiar':1,'Moderately familiar':2,'Very familiar':3,'Expert':4}
df['freq_num'] = df['GameFrequency'].map(freq_order)
df['ai_num']   = df['AIFamiliarity'].map(ai_order)
df['AgeGroup'] = pd.cut(df['Age'], bins=[17,22,30,100], labels=['18–22','23–30','30+'])
df['played_bin'] = (df['PlayedGames']=='Yes').astype(int)

# Derived condition groupings (parsed from the Condition column)
df['AIStance'] = df['Condition'].apply(lambda x: x.split()[-1] if pd.notna(x) else np.nan)   # Positive / Negative / Control
df['Openness'] = df['Condition'].apply(lambda x: x.split()[0]  if pd.notna(x) else np.nan)   # Control / Closed / Open / Neutral

BLUE   = '#3266ad'
GREEN  = '#1D9E75'
CORAL  = '#D85A30'
PINK   = '#D4537E'
AMBER  = '#BA7517'
PURPLE = '#534AB7'
GRAY   = '#888780'

def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def save(fig, name):
    fig.savefig(f'pre_plots/{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {name}.png')

# ─────────────────────────────────────────────────────────────
# PLOT 1 — Gaming Freq vs AI Familiarity
# Best type: Heatmap (two ordinal variables — shows density clearly)
# ─────────────────────────────────────────────────────────────
rho1, p1 = spearmanr(df['freq_num'].dropna(), df.loc[df['freq_num'].notna(),'ai_num'].dropna())
chi2_1, p_chi2_1, dof1, _ = chi2_contingency(
    pd.crosstab(df['GameFrequency'], df['AIFamiliarity'])
    .reindex(['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'])
    .reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
)

ct1 = pd.crosstab(df['GameFrequency'], df['AIFamiliarity'])
ct1 = ct1.reindex(['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'])
ct1 = ct1.reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
ct1.index = ['Rarely\n(few×/yr)', 'Occasionally\n(monthly)', 'Regularly\n(weekly+)']
ct1.columns = ['Slightly\nfamiliar','Moderately\nfamiliar','Very\nfamiliar','Expert']

fig, ax = plt.subplots(figsize=(7,4.5))
im = ax.imshow(ct1.values, cmap='YlGn', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(ct1.columns, fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(ct1.index, fontsize=10)
for i in range(3):
    for j in range(4):
        v = ct1.values[i,j]
        ax.text(j, i, str(v), ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if v > 5 else '#333')
plt.colorbar(im, ax=ax, label='Count')
ax.set_xlabel('AI familiarity', fontsize=11)
ax.set_ylabel('Gaming frequency', fontsize=11)
ax.set_title(f'Gaming frequency vs AI familiarity\n'
             f'Spearman ρ={rho1:.3f}, p={p1:.4f} {sig_label(p1)}  |  '
             f'χ²={chi2_1:.2f}, df={dof1}, p={p_chi2_1:.4f} {sig_label(p_chi2_1)}',
             fontsize=11, fontweight='bold', pad=12)
fig.tight_layout()
save(fig, 'plot1_gaming_vs_ai')

# ─────────────────────────────────────────────────────────────
# PLOT 2 — Gender vs Gaming Frequency
# Best type: 100% stacked bar (proportion comparison between groups)
# ─────────────────────────────────────────────────────────────
ct2 = pd.crosstab(df['Gender'], df['GameFrequency'])
ct2 = ct2.reindex(columns=['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'], fill_value=0)
ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100

chi2_2, p_chi2_2, dof2, _ = chi2_contingency(ct2)
u2, p_mw2 = mannwhitneyu(
    df[df['Gender']=='Male']['freq_num'].dropna(),
    df[df['Gender']=='Female']['freq_num'].dropna(), alternative='two-sided')

fig, ax = plt.subplots(figsize=(6,4.5))
cols_freq = ['#B5D4F4','#378ADD','#0C447C']
labels_freq = ['Rarely','Occasionally','Regularly']
bottoms = np.zeros(2)
genders = ct2_pct.index.tolist()
for i, (col, lbl, c) in enumerate(zip(ct2.columns, labels_freq, cols_freq)):
    vals = ct2_pct[col].values
    bars = ax.barh(genders, vals, left=bottoms, color=c, label=lbl, edgecolor='white', height=0.5)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 7:
            ax.text(b + v/2, j, f'{v:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    bottoms += vals

ax.set_xlim(0, 100)
ax.set_xlabel('Percentage (%)', fontsize=11)
ax.set_title(f'Gender vs gaming frequency (proportions)\n'
             f'χ²={chi2_2:.2f}, df={dof2}, p={p_chi2_2:.4f} {sig_label(p_chi2_2)}  |  '
             f'Mann-Whitney p={p_mw2:.4f} {sig_label(p_mw2)}',
             fontsize=11, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=9)
ax.set_facecolor('#f9f9f9')
fig.tight_layout()
save(fig, 'plot2_gender_vs_gaming')

# ─────────────────────────────────────────────────────────────
# PLOT 3 — Age vs Gaming Frequency  ← NEW significant relation
# Best type: Box plot (continuous vs ordinal — shows distribution)
# ─────────────────────────────────────────────────────────────
groups_age_freq = [
    df[df['GameFrequency']=='Rarely (a few times a year)']['Age'].dropna(),
    df[df['GameFrequency']=='Occasionally (monthly)']['Age'].dropna(),
    df[df['GameFrequency']=='Regularly (weekly or more)']['Age'].dropna(),
]
H, p_kw = kruskal(*groups_age_freq)

fig, ax = plt.subplots(figsize=(6,4.5))
bp = ax.boxplot(groups_age_freq, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2))
colors_bp = ['#B5D4F4','#378ADD','#0C447C']
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c)
for whisker in bp['whiskers']: whisker.set(color=GRAY, linewidth=1.2)
for cap in bp['caps']: cap.set(color=GRAY, linewidth=1.2)
for flier in bp['fliers']: flier.set(marker='o', color=GRAY, alpha=0.5, markersize=5)

ax.set_xticklabels(['Rarely\n(few×/yr)','Occasionally\n(monthly)','Regularly\n(weekly+)'], fontsize=10)
ax.set_ylabel('Age', fontsize=11)
ax.set_title(f'Age vs gaming frequency\n'
             f'Kruskal-Wallis H={H:.2f}, p={p_kw:.4f} {sig_label(p_kw)}',
             fontsize=11, fontweight='bold', pad=12)
for i, g in enumerate(groups_age_freq):
    ax.text(i+1, g.max()+1.5, f'n={len(g)}\nμ={g.mean():.1f}', ha='center', fontsize=8, color='#444')
ax.set_facecolor('#f9f9f9')
fig.tight_layout()
save(fig, 'plot3_age_vs_gaming')

# ─────────────────────────────────────────────────────────────
# PLOT 4 — Gender vs AI Familiarity
# Best type: Grouped bar with % labels
# ─────────────────────────────────────────────────────────────
ct4 = pd.crosstab(df['Gender'], df['AIFamiliarity'])
ct4 = ct4.reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
ct4_pct = ct4.div(ct4.sum(axis=1), axis=0) * 100
chi2_4, p_chi2_4, dof4, _ = chi2_contingency(ct4)
u4, p_mw4 = mannwhitneyu(
    df[df['Gender']=='Male']['ai_num'].dropna(),
    df[df['Gender']=='Female']['ai_num'].dropna(), alternative='two-sided')

fig, ax = plt.subplots(figsize=(6,4.5))
cols_ai = ['#B5D4F4','#378ADD','#0C447C','#042C53']
bottoms = np.zeros(2)
genders = ct4_pct.index.tolist()
for col, c in zip(ct4.columns, cols_ai):
    vals = ct4_pct[col].values
    ax.barh(genders, vals, left=bottoms, color=c, label=col, edgecolor='white', height=0.5)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 7:
            ax.text(b + v/2, j, f'{v:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    bottoms += vals
ax.set_xlim(0, 100)
ax.set_xlabel('Percentage (%)', fontsize=11)
ax.set_title(f'Gender vs AI familiarity (proportions)\n'
             f'χ²={chi2_4:.2f}, df={dof4}, p={p_chi2_4:.4f} {sig_label(p_chi2_4)}  |  '
             f'Mann-Whitney p={p_mw4:.4f} {sig_label(p_mw4)}',
             fontsize=11, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=8)
ax.set_facecolor('#f9f9f9')
fig.tight_layout()
save(fig, 'plot4_gender_vs_ai')

# ─────────────────────────────────────────────────────────────
# PLOT 5 — Gaming experience vs Until Dawn familiarity
# Best type: Mosaic-style grouped bar showing proportions clearly
# ─────────────────────────────────────────────────────────────
ct5 = pd.crosstab(df['PlayedGames'], df['UntilDawnFamiliarity'])
ct5 = ct5.reindex(columns=['Not familiar at all','Slightly familiar'], fill_value=0)
ct5_pct = ct5.div(ct5.sum(axis=1), axis=0) * 100
chi2_5, p_chi2_5, dof5, _ = chi2_contingency(ct5)

fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
pie_colors = ['#B5D4F4','#0C447C']
for ax, (label, row) in zip(axes, ct5_pct.iterrows()):
    wedges, texts, autotexts = ax.pie(
        row.values, labels=None, autopct='%1.0f%%',
        colors=pie_colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=12))
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight('bold'); at.set_color('white')
    n = ct5.loc[label].sum()
    ax.set_title(f'{"Gamer" if label=="Yes" else "Non-gamer"}\n(n={n})', fontsize=11, fontweight='bold')

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(pie_colors, ['Not familiar at all','Slightly familiar'])]
fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=9, frameon=False)
fig.suptitle(f'Gaming experience vs Until Dawn familiarity\n'
             f'χ²={chi2_5:.2f}, df={dof5}, p={p_chi2_5:.4f} {sig_label(p_chi2_5)}',
             fontsize=11, fontweight='bold', y=1.0)
fig.tight_layout(rect=[0,0.08,1,1])
save(fig, 'plot5_gaming_vs_untildawn')

# ─────────────────────────────────────────────────────────────
# PLOT 6 — NEW: Age vs Gaming Frequency scatter with jitter
# (shows individual data points on top of boxplot summary)
# ─────────────────────────────────────────────────────────────
# Already covered in Plot 3 (boxplot). Instead: Spearman correlation matrix heatmap
from scipy.stats import spearmanr as sr

vars_labels = ['Age','Gaming\nfreq','AI\nfamiliarity','Played\ngames','UD\nfamiliarity']
num_df = df[['Age','freq_num','ai_num','played_bin']].copy()
ud_map = {'Not familiar at all':0,'Slightly familiar':1}
num_df['ud_num'] = df['UntilDawnFamiliarity'].map(ud_map)
num_df = num_df.dropna()

n_vars = 5
rho_mat = np.zeros((n_vars, n_vars))
p_mat   = np.ones((n_vars, n_vars))
cols_num = num_df.columns.tolist()
for i in range(n_vars):
    for j in range(n_vars):
        if i != j:
            r, p = sr(num_df.iloc[:,i], num_df.iloc[:,j])
            rho_mat[i,j] = r
            p_mat[i,j]   = p
        else:
            rho_mat[i,j] = 1.0

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(rho_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Spearman ρ')
ax.set_xticks(range(n_vars)); ax.set_xticklabels(vars_labels, fontsize=10)
ax.set_yticks(range(n_vars)); ax.set_yticklabels(vars_labels, fontsize=10)
for i in range(n_vars):
    for j in range(n_vars):
        p_val = p_mat[i,j]
        sig   = sig_label(p_val) if i != j else ''
        txt   = f'{rho_mat[i,j]:.2f}\n{sig}' if i != j else '1.00'
        color = 'white' if abs(rho_mat[i,j]) > 0.5 else '#222'
        ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color, fontweight='bold')
ax.set_title('Spearman correlation matrix — all variables\n(*** p<.001  ** p<.01  * p<.05  ns = not significant)',
             fontsize=11, fontweight='bold', pad=12)
fig.tight_layout()
save(fig, 'plot6_correlation_matrix')

print('\nAll 6 pre-existing plots saved.')

# ─────────────────────────────────────────────────────────────
# PLOT 7 — Condition vs AI Familiarity
# Best type: Grouped 100% stacked bar (shows distribution within each condition)
# ─────────────────────────────────────────────────────────────
cond_order = ['Control','Closed Negative','Closed Positive','Neutral Negative','Neutral Positive','Open Negative','Open Positive']
ai_levels  = ['Slightly familiar','Moderately familiar','Very familiar','Expert']

ct7 = pd.crosstab(df['Condition'], df['AIFamiliarity'])
ct7 = ct7.reindex(cond_order, fill_value=0).reindex(columns=ai_levels, fill_value=0)
ct7_pct = ct7.div(ct7.sum(axis=1), axis=0) * 100

chi2_7, p_chi2_7, dof7, _ = chi2_contingency(ct7)
H7, p_kw7 = kruskal(*[df[df['Condition']==c]['ai_num'].dropna() for c in cond_order if len(df[df['Condition']==c]['ai_num'].dropna()) > 0])

cols_ai4 = ['#B5D4F4','#378ADD','#0C447C','#042C53']
fig, ax = plt.subplots(figsize=(8, 5))
bottoms = np.zeros(len(cond_order))
for col, c in zip(ai_levels, cols_ai4):
    vals = ct7_pct[col].values
    ax.barh(cond_order, vals, left=bottoms, color=c, label=col, edgecolor='white', height=0.55)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 8:
            ax.text(b + v/2, j, f'{v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    bottoms += vals

# Annotate n per condition
for j, cond in enumerate(cond_order):
    n = ct7.loc[cond].sum()
    ax.text(101, j, f'n={n}', va='center', fontsize=8, color='#444')

ax.set_xlim(0, 108)
ax.set_xlabel('Percentage (%)', fontsize=11)
ax.set_title(f'Condition vs AI familiarity (proportions)\n'
             f'χ²={chi2_7:.2f}, df={dof7}, p={p_chi2_7:.4f} {sig_label(p_chi2_7)}  |  '
             f'Kruskal-Wallis H={H7:.2f}, p={p_kw7:.4f} {sig_label(p_kw7)}',
             fontsize=11, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.set_facecolor('#f9f9f9')
fig.tight_layout()
save(fig, 'plot7_condition_vs_ai_familiarity')

# ─────────────────────────────────────────────────────────────
# PLOT 8 — Condition vs Gaming Frequency
# Best type: 100% stacked horizontal bar
# ─────────────────────────────────────────────────────────────
freq_levels = ['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)']
freq_labels  = ['Rarely','Occasionally','Regularly']

ct8 = pd.crosstab(df['Condition'], df['GameFrequency'])
ct8 = ct8.reindex(cond_order, fill_value=0).reindex(columns=freq_levels, fill_value=0)
ct8_pct = ct8.div(ct8.sum(axis=1), axis=0) * 100

chi2_8, p_chi2_8, dof8, _ = chi2_contingency(ct8)
H8, p_kw8 = kruskal(*[df[df['Condition']==c]['freq_num'].dropna() for c in cond_order if len(df[df['Condition']==c]['freq_num'].dropna()) > 0])

cols_freq3 = ['#B5D4F4','#378ADD','#0C447C']
fig, ax = plt.subplots(figsize=(8, 5))
bottoms = np.zeros(len(cond_order))
for col, lbl, c in zip(freq_levels, freq_labels, cols_freq3):
    vals = ct8_pct[col].values
    ax.barh(cond_order, vals, left=bottoms, color=c, label=lbl, edgecolor='white', height=0.55)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 8:
            ax.text(b + v/2, j, f'{v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    bottoms += vals

for j, cond in enumerate(cond_order):
    n = ct8.loc[cond].sum()
    ax.text(101, j, f'n={n}', va='center', fontsize=8, color='#444')

ax.set_xlim(0, 108)
ax.set_xlabel('Percentage (%)', fontsize=11)
ax.set_title(f'Condition vs gaming frequency (proportions)\n'
             f'χ²={chi2_8:.2f}, df={dof8}, p={p_chi2_8:.4f} {sig_label(p_chi2_8)}  |  '
             f'Kruskal-Wallis H={H8:.2f}, p={p_kw8:.4f} {sig_label(p_kw8)}',
             fontsize=11, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.set_facecolor('#f9f9f9')
fig.tight_layout()
save(fig, 'plot8_condition_vs_gaming_frequency')

# ─────────────────────────────────────────────────────────────
# PLOT 9 — AI Stance (Positive/Negative/Control) vs AI Familiarity
# Collapsed view: cleaner signal with more power per group
# ─────────────────────────────────────────────────────────────
stance_order = ['Control','Negative','Positive']
ct9 = pd.crosstab(df['AIStance'], df['AIFamiliarity'])
ct9 = ct9.reindex(stance_order, fill_value=0).reindex(columns=ai_levels, fill_value=0)
ct9_pct = ct9.div(ct9.sum(axis=1), axis=0) * 100
chi2_9, p_chi2_9, dof9, _ = chi2_contingency(ct9)
H9, p_kw9 = kruskal(*[df[df['AIStance']==s]['ai_num'].dropna() for s in stance_order if len(df[df['AIStance']==s]['ai_num'].dropna()) > 0])

stance_colors = ['#AAAAAA', CORAL, BLUE]   # control=gray, neg=coral, pos=blue
fig, axes = plt.subplots(1, 3, figsize=(9, 4.5), sharey=False)
for ax, (stance, color) in zip(axes, zip(stance_order, stance_colors)):
    sub = df[df['AIStance']==stance]['ai_num'].dropna()
    counts = [len(sub[sub==v]) for v in [1,2,3,4]]
    total = sum(counts)
    pcts  = [c/total*100 if total else 0 for c in counts]
    bars  = ax.bar(range(4), pcts, color=color, edgecolor='white', width=0.6, alpha=0.85)
    for b, pct in zip(bars, pcts):
        if pct > 3:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{pct:.0f}%',
                    ha='center', fontsize=10, fontweight='bold', color='#333')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Slightly','Moderately','Very','Expert'], fontsize=8, rotation=15)
    ax.set_ylim(0, 75)
    ax.set_title(f'{stance}\n(n={total})', fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel('% of group' if ax == axes[0] else '', fontsize=10)
    ax.set_facecolor('#f9f9f9')

fig.suptitle(f'AI stance (collapsed) vs AI familiarity\n'
             f'χ²={chi2_9:.2f}, df={dof9}, p={p_chi2_9:.4f} {sig_label(p_chi2_9)}  |  '
             f'Kruskal-Wallis H={H9:.2f}, p={p_kw9:.4f} {sig_label(p_kw9)}',
             fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'plot9_aistance_vs_ai_familiarity')

print('\nPlots 7–9 saved.')

# ─────────────────────────────────────────────────────────────
# PLOT 10 — Condition Valence vs Gaming Frequency  [NEW]
# Positive vs Negative conditions — are gamers distributed equally?
# Best type: Grouped bar (count comparison between two valence groups)
# ─────────────────────────────────────────────────────────────
valence_df   = df[df['Openness'] != 'Control'].copy()   # exclude control (no valence)
freq_levels  = ['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)']
short_labels = ['Rarely','Occasionally','Regularly']

ct10 = pd.crosstab(valence_df['AIStance'], valence_df['GameFrequency'])
ct10 = ct10.reindex(['Negative','Positive'], fill_value=0).reindex(columns=freq_levels, fill_value=0)

chi2_10, p_chi2_10, dof10, _ = chi2_contingency(ct10)
neg_freq = valence_df[valence_df['AIStance']=='Negative']['freq_num'].dropna()
pos_freq = valence_df[valence_df['AIStance']=='Positive']['freq_num'].dropna()
u10, p_mw10 = mannwhitneyu(neg_freq, pos_freq, alternative='two-sided') if len(neg_freq)>0 and len(pos_freq)>0 else (np.nan, np.nan)

fig, ax = plt.subplots(figsize=(6, 4.5))
x10 = np.arange(3)
for i, (valence, color) in enumerate([('Negative', CORAL), ('Positive', BLUE)]):
    vals = ct10.loc[valence].values
    bars = ax.bar(x10 + i*0.35, vals, width=0.35, label=valence, color=color,
                  edgecolor='white', alpha=0.88)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, str(int(v)),
                    ha='center', fontsize=10, fontweight='bold', color='#333')

ax.set_xticks(x10 + 0.175)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Condition valence vs gaming frequency [NEW]\n'
             f'chi2={chi2_10:.2f}, df={dof10}, p={p_chi2_10:.4f} {sig_label(p_chi2_10)}  |  '
             f'Mann-Whitney p={p_mw10:.4f} {sig_label(p_mw10)}',
             fontsize=11, fontweight='bold', pad=12)
ax.legend(title='AI Valence', fontsize=10)
ax.set_facecolor('#f9f9f9')
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
save(fig, 'plot10_valence_vs_gaming_frequency')

print('\n Done.')