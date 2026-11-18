import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# Create output directory for individual figures
output_dir = 'pre_survey_figs'
os.makedirs(output_dir, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel('Until_Dawn_Pre-survey_April_6__2026_12_17__1_.xlsx', header=1)
df.columns = ['TeamID','Condition','Age','Gender','Race','PlayedGames',
              'GameFrequency','AIFamiliarity','PlayedUntilDawn','UntilDawnFamiliarity']

# Normalize condition to Title Case (handles 'Control'/'control' inconsistency)
df['Condition'] = df['Condition'].str.strip().str.title()

# Derived condition groupings
df['ConditionType']    = df['Condition'].apply(lambda x: x.split()[0] if pd.notna(x) else np.nan)   # Control/Closed/Open/Neutral
df['ConditionValence'] = df['Condition'].apply(lambda x: x.split()[1] if pd.notna(x) and len(x.split())>1 else np.nan)  # Positive/Negative

# Ordinal encodings
freq_order = {'Rarely (a few times a year)':1, 'Occasionally (monthly)':2, 'Regularly (weekly or more)':3}
ai_order   = {'Slightly familiar':1, 'Moderately familiar':2, 'Very familiar':3, 'Expert':4}

df['freq_num'] = df['GameFrequency'].map(freq_order)
df['ai_num']   = df['AIFamiliarity'].map(ai_order)
df['AgeGroup'] = pd.cut(df['Age'], bins=[17,22,30,100], labels=['18–22','23–30','30+'])

COLORS = {'Male':'#3266ad', 'Female':'#D4537E',
          'bar1':'#1D9E75', 'bar2':'#BA7517', 'bar3':'#3266ad',
          'neutral':'#888780'}

# ── Layout: 4 rows x 2 cols (5 original + 3 new = 8 plots; Relations 1 spans full width) ──
fig = plt.figure(figsize=(18, 30))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ══════════════════════════════════════════════════════════════════════════
# RELATION 1 — Gaming Frequency vs AI Familiarity  (Spearman + grouped bar)
# ══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
gamers = df.dropna(subset=['freq_num','ai_num'])
rho, p_rho = spearmanr(gamers['freq_num'], gamers['ai_num'])

ct1  = pd.crosstab(df['GameFrequency'], df['AIFamiliarity'])
freq_labels = ['Rarely\n(few times/yr)', 'Occasionally\n(monthly)', 'Regularly\n(weekly+)']
ai_labels   = ['Slightly\nfamiliar', 'Moderately\nfamiliar', 'Very\nfamiliar', 'Expert']
ct1 = ct1.reindex(['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'])
ct1 = ct1.reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)

x   = np.arange(len(freq_labels))
w   = 0.18
bar_colors = ['#9FE1CB','#1D9E75','#0F6E56','#04342C']
for i, (col, lbl, c) in enumerate(zip(ct1.columns, ai_labels, bar_colors)):
    ax1.bar(x + i*w, ct1[col].values, width=w, label=lbl, color=c, edgecolor='white')

ax1.set_xticks(x + 1.5*w)
ax1.set_xticklabels(freq_labels, fontsize=11)
ax1.set_ylabel('Number of participants', fontsize=11)
ax1.set_title(f'Relation 1 — Gaming frequency vs AI familiarity\n'
              f'Spearman ρ = {rho:.3f},  p = {p_rho:.4f}  ({"significant" if p_rho<0.05 else "not significant"} at α=0.05)',
              fontsize=12, fontweight='bold')
ax1.legend(title='AI familiarity', fontsize=10, loc='upper left')
ax1.set_facecolor('#f9f9f9')
ax1.grid(axis='y', alpha=0.4)

chi2_1, p_chi2_1, dof1, _ = chi2_contingency(ct1)
ax1.text(0.98, 0.95,
         f'Chi-square = {chi2_1:.2f}, df={dof1}, p={p_chi2_1:.4f}',
         transform=ax1.transAxes, fontsize=9, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5f0', edgecolor='#1D9E75'))

# ══════════════════════════════════════════════════════════════════════════
# RELATION 2 — Gender vs Gaming Frequency
# ══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0])
ct2 = pd.crosstab(df['Gender'], df['GameFrequency'])
ct2 = ct2.reindex(columns=['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'], fill_value=0)
short_freq = ['Rarely', 'Occasionally', 'Regularly']
x2 = np.arange(len(short_freq))
for i, (gender, color) in enumerate([('Male', COLORS['Male']), ('Female', COLORS['Female'])]):
    vals = ct2.loc[gender].values if gender in ct2.index else [0,0,0]
    ax2.bar(x2 + i*0.35, vals, width=0.35, label=gender, color=color, edgecolor='white')

chi2_2, p_chi2_2, dof2, _ = chi2_contingency(ct2)
ax2.set_xticks(x2 + 0.175)
ax2.set_xticklabels(short_freq, fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title(f'Relation 2 — Gender vs Gaming frequency\nChi-square={chi2_2:.2f}, df={dof2}, p={p_chi2_2:.4f}',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_facecolor('#f9f9f9')
ax2.grid(axis='y', alpha=0.4)

male_freq   = df[df['Gender']=='Male']['freq_num'].dropna()
female_freq = df[df['Gender']=='Female']['freq_num'].dropna()
u2, p_mw2   = mannwhitneyu(male_freq, female_freq, alternative='two-sided')
ax2.text(0.98, 0.95, f'Mann-Whitney U={u2:.0f}, p={p_mw2:.4f}',
         transform=ax2.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef3fa', edgecolor='#3266ad'))

# ══════════════════════════════════════════════════════════════════════════
# RELATION 3 — Age vs AI Familiarity
# ══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1])
ct3 = pd.crosstab(df['AgeGroup'], df['AIFamiliarity'])
ct3 = ct3.reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
age_labels = ['18–22','23–30','30+']
x3 = np.arange(len(age_labels))
age_colors = ['#9FE1CB','#1D9E75','#0F6E56','#04342C']
for i, (col, c) in enumerate(zip(ct3.columns, age_colors)):
    ax3.bar(x3 + i*0.18, ct3[col].values, width=0.18, label=col, color=c, edgecolor='white')

chi2_3, p_chi2_3, dof3, _ = chi2_contingency(ct3)
rho3, p_rho3 = spearmanr(df['Age'].dropna(), df.loc[df['Age'].notna(),'ai_num'].dropna())
ax3.set_xticks(x3 + 0.27)
ax3.set_xticklabels(age_labels, fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title(f'Relation 3 — Age vs AI familiarity\nSpearman ρ(Age,AI)={rho3:.3f}, p={p_rho3:.4f}',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')
ax3.set_facecolor('#f9f9f9')
ax3.grid(axis='y', alpha=0.4)
ax3.text(0.98, 0.95, f'Chi-square={chi2_3:.2f}, df={dof3}, p={p_chi2_3:.4f}',
         transform=ax3.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5f0', edgecolor='#1D9E75'))

# ══════════════════════════════════════════════════════════════════════════
# RELATION 4 — Gender vs AI Familiarity
# ══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[2, 0])
ct4 = pd.crosstab(df['Gender'], df['AIFamiliarity'])
ct4 = ct4.reindex(columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
x4  = np.arange(4)
for i, (gender, color) in enumerate([('Male', COLORS['Male']), ('Female', COLORS['Female'])]):
    vals = ct4.loc[gender].values if gender in ct4.index else [0,0,0,0]
    ax4.bar(x4 + i*0.35, vals, width=0.35, label=gender, color=color, edgecolor='white')

chi2_4, p_chi2_4, dof4, _ = chi2_contingency(ct4)
male_ai   = df[df['Gender']=='Male']['ai_num'].dropna()
female_ai = df[df['Gender']=='Female']['ai_num'].dropna()
u4, p_mw4 = mannwhitneyu(male_ai, female_ai, alternative='two-sided')
ax4.set_xticks(x4 + 0.175)
ax4.set_xticklabels(['Slightly\nfam.','Moderately\nfam.','Very\nfam.','Expert'], fontsize=9)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title(f'Relation 4 — Gender vs AI familiarity\nChi-square={chi2_4:.2f}, df={dof4}, p={p_chi2_4:.4f}',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_facecolor('#f9f9f9')
ax4.grid(axis='y', alpha=0.4)
ax4.text(0.98, 0.95, f'Mann-Whitney U={u4:.0f}, p={p_mw4:.4f}',
         transform=ax4.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef3fa', edgecolor='#3266ad'))

# ══════════════════════════════════════════════════════════════════════════
# RELATION 5 — Gaming experience vs Until Dawn familiarity
# ══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, 1])
ct5 = pd.crosstab(df['PlayedGames'], df['UntilDawnFamiliarity'])
ct5 = ct5.reindex(columns=['Not familiar at all','Slightly familiar'], fill_value=0)
x5  = np.arange(2)
for i, (played, color) in enumerate([('Yes','#1D9E75'),('No','#D85A30')]):
    vals = ct5.loc[played].values if played in ct5.index else [0,0]
    ax5.bar(x5 + i*0.35, vals, width=0.35, label=f'Played games: {played}', color=color, edgecolor='white')

chi2_5, p_chi2_5, dof5, _ = chi2_contingency(ct5)
ax5.set_xticks(x5 + 0.175)
ax5.set_xticklabels(['Not familiar\nat all','Slightly\nfamiliar'], fontsize=10)
ax5.set_ylabel('Count', fontsize=10)
ax5.set_title(f'Relation 5 — Gaming experience vs Until Dawn familiarity\nChi-square={chi2_5:.2f}, df={dof5}, p={p_chi2_5:.4f}',
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=10)
ax5.set_facecolor('#f9f9f9')
ax5.grid(axis='y', alpha=0.4)

# ══════════════════════════════════════════════════════════════════════════
# RELATION 6 — Condition Type vs AI Familiarity  [NEW]
# Do Open/Closed/Neutral groups differ in prior AI familiarity?
# Kruskal-Wallis + Chi-square; stacked horizontal bars
# ══════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[3, 0])
type_order = ['Control','Closed','Neutral','Open']
ct6 = pd.crosstab(df['ConditionType'], df['AIFamiliarity'])
ct6 = ct6.reindex(type_order, fill_value=0).reindex(
    columns=['Slightly familiar','Moderately familiar','Very familiar','Expert'], fill_value=0)
ct6_pct = ct6.div(ct6.sum(axis=1), axis=0) * 100

chi2_6, p_chi2_6, dof6, _ = chi2_contingency(ct6)
kw_groups6 = [df[df['ConditionType']==t]['ai_num'].dropna() for t in type_order
              if len(df[df['ConditionType']==t]['ai_num'].dropna()) > 0]
H6, p_kw6 = kruskal(*kw_groups6)

cols_ai4 = ['#9FE1CB','#1D9E75','#0F6E56','#04342C']
bottoms6 = np.zeros(len(type_order))
for col, c in zip(ct6.columns, cols_ai4):
    vals = ct6_pct[col].values
    ax6.barh(type_order, vals, left=bottoms6, color=c, label=col, edgecolor='white', height=0.5)
    for j, (v, b) in enumerate(zip(vals, bottoms6)):
        if v > 8:
            ax6.text(b + v/2, j, f'{v:.0f}%', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
    bottoms6 += vals

for j, t in enumerate(type_order):
    n = ct6.loc[t].sum()
    ax6.text(101, j, f'n={n}', va='center', fontsize=8, color='#444')

ax6.set_xlim(0, 112)
ax6.set_xlabel('Percentage (%)', fontsize=10)
ax6.set_title(f'Relation 6 — Condition type vs AI familiarity [NEW]\n'
              f'Chi-square={chi2_6:.2f}, df={dof6}, p={p_chi2_6:.4f}  |  '
              f'Kruskal-Wallis H={H6:.2f}, p={p_kw6:.4f}',
              fontsize=11, fontweight='bold')
ax6.legend(fontsize=8, loc='lower right')
ax6.set_facecolor('#f9f9f9')
ax6.grid(axis='x', alpha=0.4)

# ══════════════════════════════════════════════════════════════════════════
# RELATION 7 — Condition Valence vs Gaming Frequency  [NEW]
# Do Positive vs Negative conditions differ in gaming frequency?
# Mann-Whitney + Chi-square; grouped bars
# ══════════════════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[3, 1])
ct7 = pd.crosstab(df['ConditionValence'], df['GameFrequency'])
ct7 = ct7.reindex(['Negative','Positive'], fill_value=0).reindex(
    columns=['Rarely (a few times a year)','Occasionally (monthly)','Regularly (weekly or more)'], fill_value=0)
short_freq7 = ['Rarely','Occasionally','Regularly']

x7 = np.arange(3)
valence_colors = ['#D85A30','#3266ad']  # Negative=coral, Positive=blue
for i, (valence, color) in enumerate([('Negative','#D85A30'),('Positive','#3266ad')]):
    vals = ct7.loc[valence].values if valence in ct7.index else [0,0,0]
    ax7.bar(x7 + i*0.35, vals, width=0.35, label=valence, color=color, edgecolor='white')

chi2_7, p_chi2_7, dof7, _ = chi2_contingency(ct7)
neg_freq = df[df['ConditionValence']=='Negative']['freq_num'].dropna()
pos_freq = df[df['ConditionValence']=='Positive']['freq_num'].dropna()
u7, p_mw7 = mannwhitneyu(neg_freq, pos_freq, alternative='two-sided') if len(neg_freq)>0 and len(pos_freq)>0 else (np.nan, np.nan)

ax7.set_xticks(x7 + 0.175)
ax7.set_xticklabels(short_freq7, fontsize=10)
ax7.set_ylabel('Count', fontsize=10)
ax7.set_title(f'Relation 7 — Condition valence vs Gaming frequency [NEW]\n'
              f'Chi-square={chi2_7:.2f}, df={dof7}, p={p_chi2_7:.4f}',
              fontsize=11, fontweight='bold')
ax7.legend(title='Valence', fontsize=10)
ax7.set_facecolor('#f9f9f9')
ax7.grid(axis='y', alpha=0.4)
ax7.text(0.98, 0.95, f'Mann-Whitney U={u7:.0f}, p={p_mw7:.4f}',
         transform=ax7.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef3fa', edgecolor='#3266ad'))

# ── Print all significance numbers to console ──────────────────────────────
print("=" * 60)
print("SIGNIFICANCE SUMMARY")
print("=" * 60)
print(f"\nRelation 1 — Gaming freq vs AI familiarity")
print(f"  Spearman ρ = {rho:.4f},  p = {p_rho:.4f}")
print(f"  Chi-square = {chi2_1:.4f}, df={dof1}, p = {p_chi2_1:.4f}")

print(f"\nRelation 2 — Gender vs Gaming frequency")
print(f"  Chi-square = {chi2_2:.4f}, df={dof2}, p = {p_chi2_2:.4f}")
print(f"  Mann-Whitney U = {u2:.0f}, p = {p_mw2:.4f}")

print(f"\nRelation 3 — Age vs AI familiarity")
print(f"  Spearman ρ(Age,AI_num) = {rho3:.4f}, p = {p_rho3:.4f}")
print(f"  Chi-square (AgeGroup vs AI) = {chi2_3:.4f}, df={dof3}, p = {p_chi2_3:.4f}")
print(f"  Mean age by AI level:")
print(df.groupby('AIFamiliarity')['Age'].mean().sort_values().to_string())

print(f"\nRelation 4 — Gender vs AI familiarity")
print(f"  Chi-square = {chi2_4:.4f}, df={dof4}, p = {p_chi2_4:.4f}")
print(f"  Mann-Whitney U = {u4:.0f}, p = {p_mw4:.4f}")

print(f"\nRelation 5 — Gaming experience vs Until Dawn familiarity")
print(f"  Chi-square = {chi2_5:.4f}, df={dof5}, p = {p_chi2_5:.4f}")
print(f"\n  Raw crosstab:")
print(ct5.to_string())

print(f"\nRelation 6 — Condition type vs AI familiarity [NEW]")
print(f"  Chi-square = {chi2_6:.4f}, df={dof6}, p = {p_chi2_6:.4f}")
print(f"  Kruskal-Wallis H = {H6:.4f}, p = {p_kw6:.4f}")
print(f"  Mean AI familiarity score by condition type:")
print(df.groupby('ConditionType')['ai_num'].mean().to_string())

print(f"\nRelation 7 — Condition valence vs Gaming frequency [NEW]")
print(f"  Chi-square = {chi2_7:.4f}, df={dof7}, p = {p_chi2_7:.4f}")
print(f"  Mann-Whitney U = {u7:.0f}, p = {p_mw7:.4f}")
print(f"  Mean gaming freq by valence:")
print(df.groupby('ConditionValence')['freq_num'].mean().to_string())

print("\n  Note: small expected cell counts may reduce chi-square reliability.")
print("=" * 60)

plt.suptitle('Until Dawn Pre-Survey — Relationship Analysis (incl. Condition)\n(n=35)',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig(os.path.join(output_dir, 'pre_survey_analysis.png'), dpi=150,
            bbox_inches='tight', facecolor='white')

print(f"\nAll plots saved to {output_dir}")
print("\nDone.")
