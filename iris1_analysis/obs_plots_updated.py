
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ─────────────────────────────────────────────
# CONFIG — change the path to your Excel file
# ─────────────────────────────────────────────
EXCEL_PATH  = "Observation_Sheet_April_6__2026_12_19.xlsx"
OUTPUT_DIR  = "obs_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & PARSE RAW DATA
# ════════════════════════════════════════════════════════════════

def clean_yn(val):
    """Convert Yes/No cell to 1/0."""
    if pd.isna(val):
        return 0
    return 1 if str(val).strip().lower() in ['yes', 'yea', 'y'] else 0


def has_ai_rec(val):
    """Return True if the AI gave a recommendation for this decision."""
    if pd.isna(val):
        return False
    return str(val).strip().lower() not in ['no', 'none', 'none ', '']


def parse_alive(val):
    """
    Parse free-text 'No. of people alive' column into an integer.
    Handles formats like:
      '2 people lived...'   → 2
      'Ashley is alive'     → 1
      'Matt and Ashley ...' → 2
    """
    if pd.isna(val):
        return np.nan
    v = str(val).lower().strip()
    if 'fate unclear' in v:
        return 1                          # conservative: only confirmed alive
    m = re.match(r'^(\d+)', v)
    if m:
        return int(m.group(1))
    names = ['ashley', 'matt', 'chris', 'emily']
    count = 0
    for name in names:
        idx = v.find(name)
        if idx == -1:
            continue
        snippet = v[idx:idx + 20]
        if 'died' not in snippet and 'dead' not in snippet:
            count += 1
    return count if count > 0 else np.nan


# Row 0 holds column labels; data starts at row 1
raw_df = pd.read_excel(EXCEL_PATH, header=0)
data   = raw_df.iloc[1:].copy().reset_index(drop=True)

DECISIONS = list(range(1, 14))   # 13 decision points per session

def parse_framing(val):
    """Parse Q3 framing condition into (FullCondition, AIStance, Openness)."""
    if pd.isna(val):
        return 'Control', 'Control', 'Control'
    v = str(val).strip()
    norm = v.lower()
    if norm in ('control',):
        return 'Control', 'Control', 'Control'
    openness = ('Open'    if 'open'    in norm else
                'Closed'  if 'closed'  in norm else
                'Neutral' if 'neutral' in norm else 'Control')
    stance   = ('Positive' if 'positive' in norm else
                'Negative' if 'negative' in norm else 'Control')
    full = f'{openness} {stance}' if openness != 'Control' else 'Control'
    return full, stance, openness


rows = []
for _, row in data.iterrows():
    team = row['Q4']          # Q4 = Team ID (Q3 was added before it)
    if pd.isna(team):
        continue

    agreements, dissents_exp, dissents_sup, alternatives, sources, ai_recs = \
        [], [], [], [], [], []

    for d in DECISIONS:
        p = f'Q5_{d}_'
        ai_recs.append(has_ai_rec(row.get(p + '2')))
        agreements.append(
            clean_yn(row.get(p + '3')) if not pd.isna(row.get(p + '3')) else 0)
        dissents_exp.append(
            clean_yn(row.get(p + '5')) if not pd.isna(row.get(p + '5')) else 0)
        dissents_sup.append(
            clean_yn(row.get(p + '6')) if not pd.isna(row.get(p + '6')) else 0)
        alternatives.append(
            clean_yn(row.get(p + '7')) if not pd.isna(row.get(p + '7')) else 0)
        src = str(row.get(p + '8', '')).strip().lower()
        if 'ai' in src and 'human' not in src:
            sources.append('AI')
        elif 'human' in src:
            sources.append('Human')
        elif 'mixed' in src:
            sources.append('Mixed')
        else:
            sources.append(None)

    n_ai  = sum(ai_recs)
    alive = parse_alive(row.get('Q7'))

    # ── Computed scores ──────────────────────────────────────
    # AI Influence  = proportion of AI-recommended decisions that were agreed with
    ai_influence = sum(agreements) / n_ai if n_ai > 0 else 0

    # Groupthink    = (agreements + suppressions − alternatives) / 13  [0–1 clipped]
    groupthink = max(0, min(1,
        (sum(agreements) + sum(dissents_sup) - sum(alternatives)) / 13))

    # Cohesion      = 1 − (dissent expressed / 13)   [high = no one spoke up]
    cohesion = 1 - sum(dissents_exp) / 13

    # Regret        = deaths / 4   (max 4 characters can die)
    regret = (4 - alive) / 4 if (alive is not None and not np.isnan(alive)) else np.nan

    # Behavior      = did the group mostly follow or reject AI?
    ai_src = sum(1 for s in sources if s == 'AI')
    hu_src = sum(1 for s in sources if s == 'Human')
    if ai_src > hu_src:
        behavior = 'Agreed'
    elif hu_src > ai_src:
        behavior = 'Disagreed'
    else:
        behavior = 'Mixed'

    # Momentum      = consecutive AI agreements at session start
    consec = next((i for i, a in enumerate(agreements) if a == 0), len(agreements))

    # Parse framing condition directly from Q3
    full_cond, stance, openness_label = parse_framing(row.get('Q3'))

    rows.append({
        'Team'          : str(team).strip(),
        'AI_Influence'  : ai_influence,
        'Groupthink'    : groupthink,
        'Cohesion'      : cohesion,
        'Regret'        : regret,
        'N_AI_Recs'     : n_ai,
        'N_Dissent_Exp' : sum(dissents_exp),
        'N_Dissent_Sup' : sum(dissents_sup),
        'N_Alternatives': sum(alternatives),
        'Consec_Agree'  : consec,
        'Behavior'      : behavior,
        'Condition'     : 'AI Present' if n_ai > 0 else 'No AI',
        'FullCondition' : full_cond,
        'AIStance'      : stance,
        'Openness'      : openness_label,
    })

df_obs = pd.DataFrame(rows)

# Average the two observers per team
def modal(x):
    return x.mode()[0]

dt = df_obs.groupby('Team').agg({
    'AI_Influence'  : 'mean',
    'Groupthink'    : 'mean',
    'Cohesion'      : 'mean',
    'Regret'        : 'mean',
    'N_AI_Recs'     : 'mean',
    'N_Dissent_Exp' : 'mean',
    'N_Dissent_Sup' : 'mean',
    'N_Alternatives': 'mean',
    'Consec_Agree'  : 'mean',
    'Behavior'      : modal,
    'Condition'     : modal,
    'FullCondition' : modal,
    'AIStance'      : modal,
    'Openness'      : modal,
}).reset_index()

dt['Valence_Score']  = dt['AIStance'].map({'Positive': 1, 'Negative': -1, 'Control': 0})
dt['Openness_Score'] = dt['Openness'].map({'Open': 1, 'Neutral': 0, 'Closed': -1, 'Control': 0})

print(f"Teams in analysis: {len(dt)}")


# ════════════════════════════════════════════════════════════════
# STEP 2 — SHARED STYLE HELPERS
# ════════════════════════════════════════════════════════════════

BG   = '#0F0F1A'
CARD = '#161628'
TEXT = '#E8E8F0'
SUB  = '#8888A8'
GRID = '#252538'

PAL = {
    'Agreed'    : '#7C5CFC',
    'Mixed'     : '#F7971E',
    'Disagreed' : '#00C9FF',
    'AI Present': '#7C5CFC',
    'No AI'     : '#43E97B',
}


def base_fig(w=7, h=5.5):
    """Create a dark-themed figure + axes."""
    plt.rcParams.update({
        'font.family'    : 'DejaVu Sans',
        'text.color'     : TEXT,
        'axes.labelcolor': TEXT,
        'xtick.color'    : TEXT,
        'ytick.color'    : TEXT,
    })
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def add_regression(ax, x, y, color='#FC5C7D', lw=2):
    """Overlay a dashed regression line + r/p annotation."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < 4:
        return
    slope, intercept, r, p, _ = stats.linregress(xc, yc)
    xl = np.linspace(xc.min(), xc.max(), 200)
    ax.plot(xl, slope * xl + intercept, '--', color=color, lw=lw, alpha=0.9, zorder=4)
    ps = ('p<.001' if p < 0.001 else
          f'p={p:.3f}' if p < 0.05 else
          f'p={p:.2f}')
    ax.text(0.97, 0.05, f'r = {r:.2f},  {ps}',
            transform=ax.transAxes, fontsize=10, color=color,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=color, alpha=0.9))


def save_fig(fig, name):
    """Tight-save to OUTPUT_DIR."""
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'  saved  {path}')


# ════════════════════════════════════════════════════════════════
# STEP 3 — SIGNIFICANCE SUMMARY (printed to console)
# ════════════════════════════════════════════════════════════════

def pearson_report(label, x, y):
    pair = pd.DataFrame({'x': x, 'y': y}).dropna()
    n = len(pair)
    r, p = stats.pearsonr(pair['x'], pair['y'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  {label:<48} r={r:+.3f}  p={p:.4f}  n={n}  {sig}")


print("\n" + "=" * 65)
print("SIGNIFICANCE TESTS")
print("=" * 65)
pearson_report("G1  AI_Influence  →  Groupthink",       dt['AI_Influence'],   dt['Groupthink'])
pearson_report("G2  Groupthink    →  Regret",            dt['Groupthink'],     dt['Regret'])
pearson_report("G5  Cohesion      →  Groupthink",        dt['Cohesion'],       dt['Groupthink'])
pearson_report("B1  N_AI_Recs     →  Groupthink",        dt['N_AI_Recs'],      dt['Groupthink'])
pearson_report("B2  N_Alternatives→  Groupthink",        dt['N_Alternatives'], dt['Groupthink'])
pearson_report("B3  Consec_Agree  →  Groupthink",        dt['Consec_Agree'],   dt['Groupthink'])
pearson_report("B4a N_Dissent_Exp →  Groupthink",        dt['N_Dissent_Exp'],  dt['Groupthink'])
pearson_report("B4b N_Dissent_Sup →  Groupthink",        dt['N_Dissent_Sup'],  dt['Groupthink'])
pearson_report("F1  Valence_Score  →  Groupthink",       dt['Valence_Score'],  dt['Groupthink'])
pearson_report("F2  Valence_Score  →  AI_Influence",     dt['Valence_Score'],  dt['AI_Influence'])
pearson_report("F3  Valence_Score  →  Regret",           dt['Valence_Score'],  dt['Regret'])
pearson_report("F4  Openness_Score →  Groupthink",       dt['Openness_Score'], dt['Groupthink'])
pearson_report("F5  Openness_Score →  AI_Influence",     dt['Openness_Score'], dt['AI_Influence'])
pearson_report("F6  Openness_Score →  Regret",           dt['Openness_Score'], dt['Regret'])

ag = dt[dt['Behavior'] == 'Agreed']['Groupthink'].values
dg = dt[dt['Behavior'] == 'Disagreed']['Groupthink'].values
t, p = stats.ttest_ind(ag, dg)
print(f"\n  G3  Agreed vs Disagreed groupthink  "
      f"t={t:.3f}  p={p:.4f}  {'***' if p<0.001 else 'ns'}")

ai_g   = dt[dt['Condition'] == 'AI Present']['Groupthink'].values
noai_g = dt[dt['Condition'] == 'No AI']['Groupthink'].values
t, p = stats.ttest_ind(ai_g, noai_g)
print(f"  G4  AI Present vs No AI groupthink  "
      f"t={t:.3f}  p={p:.4f}  {'**' if p<0.01 else 'ns'}")

ar = dt[dt['Behavior'] == 'Agreed']['Regret'].dropna()
dr = dt[dt['Behavior'] == 'Disagreed']['Regret'].dropna()
t, p = stats.ttest_ind(ar, dr)
print(f"  B5  Agreed vs Disagreed regret       "
      f"t={t:.3f}  p={p:.4f}  ns  (Agreed={ar.mean():.3f} Disagreed={dr.mean():.3f})")
print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════════
# STEP 4 — INDIVIDUAL PLOTS
# ════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)   # fixed seed for reproducible jitter

# ── Plot 01 · AI Influence → Groupthink ─────────────────────────
# Best plot type: scatter + regression line
# Continuous x → continuous y with clear linear trend.
fig, ax = base_fig()
ax.set_title('Graph 1 · AI Influence → Groupthink',
             fontsize=14, fontweight='bold', pad=12, color=TEXT)
sc = ax.scatter(dt['AI_Influence'], dt['Groupthink'],
                c=dt['Groupthink'], cmap='plasma', s=90,
                alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt['AI_Influence'].values, dt['Groupthink'].values)
for _, row in dt.iterrows():
    if row['Groupthink'] > 0.75 or row['AI_Influence'] > 0.95:
        ax.annotate(f"T{row['Team']}",
                    (row['AI_Influence'], row['Groupthink']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color=SUB)
ax.set_xlabel('AI Influence Score  (proportion of decisions agreed with AI)', fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(colors=SUB, labelsize=9)
save_fig(fig, '01_ai_influence_vs_groupthink')

# ── Plot 02 · Groupthink → Regret ───────────────────────────────
# Best plot type: bubble scatter (3rd variable = N_AI_Recs as size)
# Shows null result clearly; bubble size adds a 3rd dimension.
fig, ax = base_fig()
ax.set_title('Graph 2 · Groupthink → Regret\n(bubble size = no. of AI recommendations)',
             fontsize=13, fontweight='bold', pad=12, color=TEXT)
dt_r   = dt.dropna(subset=['Regret'])
sizes  = (dt_r['N_AI_Recs'] + 1) * 18
colors_r = [PAL.get(b, '#888') for b in dt_r['Behavior']]
ax.scatter(dt_r['Groupthink'], dt_r['Regret'],
           s=sizes, c=colors_r, alpha=0.75,
           edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt_r['Groupthink'].values, dt_r['Regret'].values, color='#F7971E')
ax.set_xlabel('Groupthink Score', fontsize=11)
ax.set_ylabel('Regret Score  (deaths ÷ 4)', fontsize=11)
legend_els = [
    mpatches.Patch(fc=PAL[b], ec='white', lw=0.5, label=b)
    for b in ['Agreed', 'Mixed', 'Disagreed']
] + [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
           markersize=7,  label='5 AI recs'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
           markersize=11, label='10 AI recs'),
]
ax.legend(handles=legend_els, frameon=True, facecolor=BG, edgecolor=GRID,
          labelcolor=TEXT, fontsize=9)
save_fig(fig, '02_groupthink_vs_regret')

# ── Plot 03 · Behavior vs Groupthink ────────────────────────────
# Best plot type: violin + strip  (better than box — shows distribution shape)
fig, ax = base_fig()
ax.set_title('Graph 3 · Behavior vs Groupthink  (violin + strip)',
             fontsize=14, fontweight='bold', pad=12, color=TEXT)
order    = ['Agreed', 'Mixed', 'Disagreed']
beh_data = [dt[dt['Behavior'] == b]['Groupthink'].values for b in order]
vp = ax.violinplot(beh_data, positions=[1, 2, 3],
                   showmedians=True, showextrema=True, widths=0.55)
for body, b in zip(vp['bodies'], order):
    body.set_facecolor(PAL[b]); body.set_alpha(0.5); body.set_edgecolor('white')
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2.5)
for part in ['cbars', 'cmins', 'cmaxes']:
    vp[part].set_color(SUB)
for i, (bd, b) in enumerate(zip(beh_data, order)):
    jitter = rng.normal(0, 0.06, len(bd))
    ax.scatter(np.full(len(bd), i + 1) + jitter, bd,
               color=PAL[b], s=55, alpha=0.85, zorder=5,
               edgecolors='white', linewidth=0.4)
    ax.text(i + 1, -0.07, f'n={len(bd)}\nmean={bd.mean():.2f}',
            ha='center', fontsize=9, color=SUB,
            transform=ax.get_xaxis_transform())
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
ax.set_xlabel('Decision behavior relative to AI', fontsize=11)
t, p = stats.ttest_ind(beh_data[0], beh_data[2])
y_top = max(bd.max() for bd in beh_data) + 0.05
ax.plot([1, 3], [y_top, y_top], color='white', lw=1)
ps = 'p<.001' if p < 0.001 else f'p={p:.3f}'
ax.text(2, y_top + 0.01, f'Agreed vs Disagreed:  {ps}',
        ha='center', fontsize=9, color='white')
save_fig(fig, '03_behavior_vs_groupthink')

# ── Plot 04 · AI Condition vs Groupthink ────────────────────────
# Best plot type: raincloud (half-histogram + strip + mean line)
# Bar chart hides the zero-variance in the No AI group.
fig, ax = base_fig(w=6, h=5)
ax.set_title('Graph 4 · AI Condition vs Groupthink  (raincloud)',
             fontsize=14, fontweight='bold', pad=12, color=TEXT)
conds  = ['AI Present', 'No AI']
colors = [PAL['AI Present'], PAL['No AI']]
for i, (cond, col) in enumerate(zip(conds, colors)):
    vals = dt[dt['Condition'] == cond]['Groupthink'].values
    # half-histogram violin on the left side
    bins = np.linspace(0, 1, 15)
    counts, edges = np.histogram(vals, bins=bins)
    scale = 0.35 / counts.max() if counts.max() > 0 else 1
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        if count == 0:
            continue
        ax.barh((lo + hi) / 2, -count * scale, height=(hi - lo) * 0.9,
                left=i + 1, color=col, alpha=0.5, edgecolor='none')
    # strip dots on the right
    jitter = rng.normal(0, 0.05, len(vals))
    ax.scatter(np.full(len(vals), i + 1 + 0.18) + jitter, vals,
               color=col, s=55, alpha=0.85, zorder=5,
               edgecolors='white', linewidth=0.4)
    # mean line
    ax.plot([i + 0.65, i + 1.35], [vals.mean(), vals.mean()],
            color='white', lw=2.5, zorder=6)
    ax.text(i + 1, -0.06,
            f'n={len(vals)}  mean={vals.mean():.2f}',
            ha='center', fontsize=9.5, color=SUB,
            transform=ax.get_xaxis_transform())
ax.set_xticks([1, 2]); ax.set_xticklabels(conds, fontsize=12)
ax.set_ylabel('Groupthink Score', fontsize=11)
ax.set_xlim(0.4, 2.6)
t, p = stats.ttest_ind(
    dt[dt['Condition'] == 'AI Present']['Groupthink'].values,
    dt[dt['Condition'] == 'No AI']['Groupthink'].values)
ps = ('p<.001' if p < 0.001 else f'p={p:.3f}' if p < 0.05 else f'p={p:.2f}')
ax.text(0.5, 0.95, f't = {t:.2f},  {ps}', transform=ax.transAxes,
        fontsize=10, color='white', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '04_condition_vs_groupthink')

# ── Plot 05 · Cohesion → Groupthink ─────────────────────────────
# Best plot type: scatter coloured by behavior — reveals why r is near-zero
fig, ax = base_fig()
ax.set_title('Graph 5 · Cohesion → Groupthink',
             fontsize=14, fontweight='bold', pad=12, color=TEXT)
for b in ['Agreed', 'Mixed', 'Disagreed']:
    sub = dt[dt['Behavior'] == b]
    ax.scatter(sub['Cohesion'], sub['Groupthink'],
               label=b, color=PAL[b], s=80, alpha=0.85,
               edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt['Cohesion'].values, dt['Groupthink'].values, color='#FC5C7D')
ax.set_xlabel('Cohesion Score  (1 = zero dissent expressed)', fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
ax.legend(title='Behavior', frameon=True, facecolor=BG, edgecolor=GRID,
          labelcolor=TEXT, title_fontsize=9, fontsize=9)
save_fig(fig, '05_cohesion_vs_groupthink')

# ── Plot 06 · Correlation Heatmap ───────────────────────────────
# Shows r + significance stars for 7 key variables
fig, ax = base_fig(w=7.5, h=6.5)
ax.set_title('Graph 6 · Correlation Heatmap  (r  /  significance)',
             fontsize=14, fontweight='bold', pad=12, color=TEXT)
h_cols   = ['AI_Influence', 'Cohesion', 'Groupthink', 'Regret',
            'N_AI_Recs', 'N_Alternatives', 'N_Dissent_Sup',
            'Valence_Score', 'Openness_Score']
h_labels = ['AI\nInfluence', 'Cohesion', 'Groupthink', 'Regret',
            'No. of AI\nRecs', 'Alternatives\nDiscussed', 'Dissent\nSuppressed',
            'Valence\n(+pos/−neg)', 'Openness\n(+open/−closed)']
n_h = len(h_cols)
r_mat = np.zeros((n_h, n_h))
p_mat = np.ones((n_h, n_h))
for i in range(n_h):
    for j in range(n_h):
        if i == j:
            r_mat[i, j] = 1.0
            continue
        pair = dt[[h_cols[i], h_cols[j]]].dropna()
        if len(pair) > 3:
            r, p = stats.pearsonr(pair[h_cols[i]], pair[h_cols[j]])
            r_mat[i, j] = r
            p_mat[i, j] = p
im = ax.imshow(r_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_h)); ax.set_yticks(range(n_h))
ax.set_xticklabels(h_labels, fontsize=9.5)
ax.set_yticklabels(h_labels, fontsize=9.5)
ax.tick_params(colors=TEXT)
for i in range(n_h):
    for j in range(n_h):
        r = r_mat[i, j]; p = p_mat[i, j]
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else '')
        txt_col = 'white' if abs(r) > 0.45 else TEXT
        ax.text(j, i, f'{r:.2f}\n{sig}',
                ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=txt_col)
cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(colors=SUB, labelsize=9)
save_fig(fig, '06_correlation_heatmap')

# ── Plot 07 · AI Exposure → Groupthink ──────────────────────────
# New finding: even teams that disagreed show higher GT with more recs
fig, ax = base_fig()
ax.set_title('Bonus 1 · AI Exposure → Groupthink\n'
             '(more AI prompts = higher groupthink even in disagreeing teams)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
for b in ['Agreed', 'Mixed', 'Disagreed']:
    sub = dt[dt['Behavior'] == b]
    ax.scatter(sub['N_AI_Recs'], sub['Groupthink'],
               label=b, color=PAL[b], s=80, alpha=0.85,
               edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt['N_AI_Recs'].values, dt['Groupthink'].values, color='#FC5C7D')
ax.set_xlabel('Number of AI Recommendations Encountered', fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
ax.legend(title='Behavior', frameon=True, facecolor=BG, edgecolor=GRID,
          labelcolor=TEXT, title_fontsize=9, fontsize=9)
save_fig(fig, '07_AI_exposure_vs_groupthink')

# ── Plot 08 · Alternatives Discussed → Groupthink ───────────────
# New finding: deliberating protects against groupthink
fig, ax = base_fig()
ax.set_title('Bonus 2 · Alternatives Discussed → Groupthink\n'
             '(deliberating protects, even if you ultimately follow AI)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
sc = ax.scatter(dt['N_Alternatives'], dt['Groupthink'],
                c=dt['AI_Influence'], cmap='cool', s=85,
                alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt['N_Alternatives'].values, dt['Groupthink'].values, color='#43E97B')
ax.set_xlabel('Number of Decisions Where Alternatives Were Discussed', fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(colors=SUB, labelsize=9)
cb.set_label('AI Influence', color=SUB, fontsize=10)
save_fig(fig, '08_alternatives_vs_groupthink')

# ── Plot 09 · Decision Momentum → Groupthink ────────────────────
# New finding: early agreement locks in conformity
fig, ax = base_fig()
ax.set_title('Bonus 3 · Decision Momentum → Groupthink\n'
             '(early consecutive agreement predicts full-session conformity)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
sc = ax.scatter(dt['Consec_Agree'], dt['Groupthink'],
                c=dt['AI_Influence'], cmap='plasma', s=85,
                alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
add_regression(ax, dt['Consec_Agree'].values, dt['Groupthink'].values, color='#FC5C7D')
ax.set_xlabel('Consecutive AI Agreements at Session Start', fontsize=11)
ax.set_ylabel('Groupthink Score', fontsize=11)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(colors=SUB, labelsize=9)
cb.set_label('AI Influence', color=SUB, fontsize=10)
save_fig(fig, '09_momentum_vs_groupthink')

# ── Plot 10 · Dissent Expressed vs Suppressed per Team ──────────
# New finding: which teams silenced dissenters
fig, ax = base_fig(w=13, h=5)
ax.set_title('Bonus 4 · Dissent Expressed vs Suppressed per Team\n'
             '(sorted low → high Groupthink — red bars = voices silenced)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
dt_s = dt.sort_values('Groupthink').reset_index(drop=True)
x = np.arange(len(dt_s))
ax.bar(x, dt_s['N_Dissent_Exp'], color='#00C9FF', alpha=0.7,
       label='Expressed', width=0.7)
ax.bar(x, dt_s['N_Dissent_Sup'], color='#FC5C7D', alpha=0.9,
       label='Suppressed', width=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f"T{t}" for t in dt_s['Team']], rotation=90, fontsize=7.5)
ax.set_xlabel('Teams  (sorted low → high Groupthink)', fontsize=11)
ax.set_ylabel('Decision count', fontsize=11)
ax.legend(frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save_fig(fig, '10_dissent_expressed_vs_suppressed')

# ── Plot 11 · Behavior vs Regret ────────────────────────────────
# Key null result: following AI did NOT lead to better outcomes
fig, ax = base_fig(w=6, h=5)
ax.set_title('Bonus 5 · Behavior vs Regret  (95% CI)\n'
             '(no significant difference — following AI ≠ better outcomes)',
             fontsize=11, fontweight='bold', pad=12, color=TEXT)
for i, b in enumerate(order):
    vals = dt[dt['Behavior'] == b]['Regret'].dropna().values
    mean = vals.mean()
    ci95 = (stats.t.ppf(0.975, df=len(vals) - 1) * stats.sem(vals)
            if len(vals) > 1 else 0)
    ax.errorbar(i + 1, mean, yerr=ci95, fmt='o', color=PAL[b],
                markersize=11, capsize=7, capthick=2, linewidth=2, zorder=6)
    ax.scatter(np.full(len(vals), i + 1) + rng.normal(0, 0.07, len(vals)),
               vals, color=PAL[b], s=35, alpha=0.55, zorder=4, edgecolors='none')
    ax.text(i + 1, -0.04, f'n={len(vals)}\nmean={mean:.2f}',
            ha='center', fontsize=9, color=SUB,
            transform=ax.get_xaxis_transform())
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel('Regret Score  (deaths ÷ 4)', fontsize=11)
ax.set_xlabel('Behavior relative to AI', fontsize=11)
ax.set_ylim(0.1, 1.0)
t, p = stats.ttest_ind(
    dt[dt['Behavior'] == 'Agreed']['Regret'].dropna(),
    dt[dt['Behavior'] == 'Disagreed']['Regret'].dropna())
ax.text(0.5, 0.95, f'Agreed vs Disagreed:  p = {p:.2f}  (ns)',
        transform=ax.transAxes, fontsize=10, color=SUB,
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '11_behavior_vs_regret')


print(f"\nDone — original 11 plots saved to '{OUTPUT_DIR}/'")

# ════════════════════════════════════════════════════════════════
# NEW CONDITION PLOTS (12 – 14)
# ════════════════════════════════════════════════════════════════

COND_ORDER = ['Control','Closed Negative','Closed Positive',
              'Neutral Negative','Neutral Positive',
              'Open Negative','Open Positive']

COND_PAL = {
    'Control'         : '#888888',
    'Closed Negative' : '#D85A30',
    'Closed Positive' : '#3266AD',
    'Neutral Negative': '#BA7517',
    'Neutral Positive': '#1D9E75',
    'Open Negative'   : '#D4537E',
    'Open Positive'   : '#534AB7',
}

# ── Plot 12 · Full Condition vs Groupthink ───────────────────────
# Best type: dot plot with CI bars — cleanly shows mean ± 95% CI per condition
fig, ax = base_fig(w=8, h=5.5)
ax.set_title('Plot 12 · Condition vs Groupthink  (mean ± 95 % CI)',
             fontsize=13, fontweight='bold', pad=12, color=TEXT)

y_pos = list(range(len(COND_ORDER)))
for yi, cond in enumerate(COND_ORDER):
    sub = dt[dt['FullCondition'] == cond]['Groupthink'].dropna().values
    if len(sub) == 0:
        continue
    mean = sub.mean()
    ci   = (stats.t.ppf(0.975, df=len(sub)-1) * stats.sem(sub)
            if len(sub) > 1 else 0)
    col  = COND_PAL[cond]
    # jitter strip
    ax.scatter(sub + rng.normal(0, 0.008, len(sub)),
               np.full(len(sub), yi), color=col, s=45,
               alpha=0.55, zorder=4, edgecolors='none')
    # CI bar
    ax.errorbar(mean, yi, xerr=ci, fmt='o', color=col,
                markersize=9, capsize=5, capthick=2, linewidth=2, zorder=6,
                ecolor=col)
    ax.text(1.02, yi, f'n={len(sub)}  μ={mean:.2f}',
            va='center', fontsize=8.5, color=SUB,
            transform=ax.get_yaxis_transform())

ax.set_yticks(y_pos)
ax.set_yticklabels(COND_ORDER, fontsize=10)
ax.set_xlabel('Groupthink Score', fontsize=11)
ax.set_xlim(-0.05, 1.05)

# Kruskal-Wallis across all conditions
groups_kw = [dt[dt['FullCondition']==c]['Groupthink'].dropna().values
             for c in COND_ORDER if len(dt[dt['FullCondition']==c]['Groupthink'].dropna()) > 0]
if len(groups_kw) >= 2:
    H12, p12 = stats.kruskal(*groups_kw)
    sig12 = '***' if p12<0.001 else '**' if p12<0.01 else '*' if p12<0.05 else 'ns'
    ax.text(0.98, 0.98, f'Kruskal-Wallis  H={H12:.2f},  p={p12:.3f}  {sig12}',
            transform=ax.transAxes, fontsize=9, color=SUB,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '12_condition_vs_groupthink')


# ── Plot 13 · Full Condition vs AI Influence ─────────────────────
# Best type: same dot-plot style for direct comparison with Plot 12
fig, ax = base_fig(w=8, h=5.5)
ax.set_title('Plot 13 · Condition vs AI Influence  (mean ± 95 % CI)',
             fontsize=13, fontweight='bold', pad=12, color=TEXT)

for yi, cond in enumerate(COND_ORDER):
    sub = dt[dt['FullCondition'] == cond]['AI_Influence'].dropna().values
    if len(sub) == 0:
        continue
    mean = sub.mean()
    ci   = (stats.t.ppf(0.975, df=len(sub)-1) * stats.sem(sub)
            if len(sub) > 1 else 0)
    col  = COND_PAL[cond]
    ax.scatter(sub + rng.normal(0, 0.008, len(sub)),
               np.full(len(sub), yi), color=col, s=45,
               alpha=0.55, zorder=4, edgecolors='none')
    ax.errorbar(mean, yi, xerr=ci, fmt='o', color=col,
                markersize=9, capsize=5, capthick=2, linewidth=2, zorder=6,
                ecolor=col)
    ax.text(1.02, yi, f'n={len(sub)}  μ={mean:.2f}',
            va='center', fontsize=8.5, color=SUB,
            transform=ax.get_yaxis_transform())

ax.set_yticks(y_pos)
ax.set_yticklabels(COND_ORDER, fontsize=10)
ax.set_xlabel('AI Influence Score', fontsize=11)
ax.set_xlim(-0.05, 1.05)

groups_kw13 = [dt[dt['FullCondition']==c]['AI_Influence'].dropna().values
               for c in COND_ORDER if len(dt[dt['FullCondition']==c]['AI_Influence'].dropna()) > 0]
if len(groups_kw13) >= 2:
    H13, p13 = stats.kruskal(*groups_kw13)
    sig13 = '***' if p13<0.001 else '**' if p13<0.01 else '*' if p13<0.05 else 'ns'
    ax.text(0.98, 0.98, f'Kruskal-Wallis  H={H13:.2f},  p={p13:.3f}  {sig13}',
            transform=ax.transAxes, fontsize=9, color=SUB,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '13_condition_vs_ai_influence')


# ── Plot 14 · AI Stance (collapsed) vs Groupthink + AI Influence ─
# Best type: side-by-side violin — Positive vs Negative vs Control
# collapsed across openness level for statistical power
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=BG)
stance_order = ['Control', 'Negative', 'Positive']
stance_pal   = {'Control': '#888888', 'Negative': '#D85A30', 'Positive': '#3266AD'}
metrics      = [('Groupthink', 'Groupthink Score'), ('AI_Influence', 'AI Influence Score')]

for ax, (metric, ylabel) in zip(axes, metrics):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=10)

    groups_data = [dt[dt['AIStance'] == s][metric].dropna().values for s in stance_order]
    vp = ax.violinplot(groups_data, positions=[1,2,3],
                       showmedians=True, showextrema=True, widths=0.55)
    for body, s in zip(vp['bodies'], stance_order):
        body.set_facecolor(stance_pal[s])
        body.set_alpha(0.5)
        body.set_edgecolor('white')
    vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2.5)
    for part in ['cbars','cmins','cmaxes']:
        vp[part].set_color(SUB)

    for i, (gd, s) in enumerate(zip(groups_data, stance_order)):
        jitter = rng.normal(0, 0.06, len(gd))
        ax.scatter(np.full(len(gd), i+1) + jitter, gd,
                   color=stance_pal[s], s=55, alpha=0.85, zorder=5,
                   edgecolors='white', linewidth=0.4)
        ax.text(i+1, -0.07,
                f'n={len(gd)}\nμ={gd.mean():.2f}' if len(gd) else f'n=0',
                ha='center', fontsize=9, color=SUB,
                transform=ax.get_xaxis_transform())

    ax.set_xticks([1,2,3]); ax.set_xticklabels(stance_order, fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
    ax.set_title(f'AI Stance vs {ylabel.split()[0]}',
                 fontsize=12, fontweight='bold', pad=10, color=TEXT)

    # Kruskal-Wallis
    valid = [g for g in groups_data if len(g) > 0]
    if len(valid) >= 2:
        H14, p14 = stats.kruskal(*valid)
        sig14 = '***' if p14<0.001 else '**' if p14<0.01 else '*' if p14<0.05 else 'ns'
        ax.text(0.97, 0.97, f'H={H14:.2f},  p={p14:.3f}  {sig14}',
                transform=ax.transAxes, fontsize=9, color=SUB,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=GRID, alpha=0.9))

fig.suptitle('Plot 14 · AI Stance (collapsed) vs Key Outcomes',
             fontsize=13, fontweight='bold', color=TEXT, y=1.01)
fig.tight_layout()
path14 = os.path.join(OUTPUT_DIR, '14_aistance_vs_outcomes.png')
fig.savefig(path14, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'  saved  {path14}')

print(f"\nDone — all 14 plots saved to '{OUTPUT_DIR}/' (11 original + 3 condition-related)")

# ════════════════════════════════════════════════════════════════
# NEW FRAMING RELATION PLOTS (15 – 19)
# ════════════════════════════════════════════════════════════════

# ── Plot 15 · Full Condition vs Regret ──────────────────────────
fig, ax = base_fig(w=8, h=5.5)
ax.set_title('Plot 15 · Condition vs Regret  (mean ± 95 % CI)\n'
             '(does framing affect how many characters die?)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
for yi, cond in enumerate(COND_ORDER):
    sub = dt[dt['FullCondition'] == cond]['Regret'].dropna().values
    if len(sub) == 0:
        continue
    mean = sub.mean()
    ci   = (stats.t.ppf(0.975, df=len(sub)-1) * stats.sem(sub)
            if len(sub) > 1 else 0)
    col  = COND_PAL[cond]
    ax.scatter(sub + rng.normal(0, 0.008, len(sub)),
               np.full(len(sub), yi), color=col, s=45,
               alpha=0.55, zorder=4, edgecolors='none')
    ax.errorbar(mean, yi, xerr=ci, fmt='o', color=col,
                markersize=9, capsize=5, capthick=2, linewidth=2, zorder=6,
                ecolor=col)
    ax.text(1.02, yi, f'n={len(sub)}  μ={mean:.2f}',
            va='center', fontsize=8.5, color=SUB,
            transform=ax.get_yaxis_transform())
ax.set_yticks(list(range(len(COND_ORDER))))
ax.set_yticklabels(COND_ORDER, fontsize=10)
ax.set_xlabel('Regret Score  (deaths ÷ 4)', fontsize=11)
ax.set_xlim(-0.05, 1.15)
groups_kw15 = [dt[dt['FullCondition']==c]['Regret'].dropna().values
               for c in COND_ORDER if len(dt[dt['FullCondition']==c]['Regret'].dropna()) >= 2]
if len(groups_kw15) >= 2:
    H15, p15 = stats.kruskal(*groups_kw15)
    sig15 = '***' if p15<0.001 else '**' if p15<0.01 else '*' if p15<0.05 else 'ns'
    ax.text(0.98, 0.98, f'Kruskal-Wallis  H={H15:.2f},  p={p15:.3f}  {sig15}',
            transform=ax.transAxes, fontsize=9, color=SUB,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '15_condition_vs_regret')


# ── Plot 16 · Openness (collapsed) vs Key Outcomes ──────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=BG)
open_order = ['Control', 'Closed', 'Neutral', 'Open']
open_pal   = {'Control': '#888888', 'Closed': '#D85A30',
              'Neutral': '#BA7517', 'Open': '#534AB7'}
metrics16  = [('Groupthink', 'Groupthink Score'), ('Regret', 'Regret Score (deaths÷4)')]
for ax, (metric, ylabel) in zip(axes, metrics16):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=10)
    groups_data16 = [dt[dt['Openness']==o][metric].dropna().values for o in open_order]
    valid16 = [(g, o) for g, o in zip(groups_data16, open_order) if len(g) >= 2]
    pos16   = list(range(1, len(valid16)+1))
    vp16 = ax.violinplot([g for g, _ in valid16], positions=pos16,
                         showmedians=True, showextrema=True, widths=0.55)
    for body, (_, o) in zip(vp16['bodies'], valid16):
        body.set_facecolor(open_pal[o]); body.set_alpha(0.5); body.set_edgecolor('white')
    vp16['cmedians'].set_color('white'); vp16['cmedians'].set_linewidth(2.5)
    for part in ['cbars','cmins','cmaxes']: vp16[part].set_color(SUB)
    for i, (gd, o) in enumerate(valid16):
        jitter = rng.normal(0, 0.06, len(gd))
        ax.scatter(np.full(len(gd), i+1)+jitter, gd,
                   color=open_pal[o], s=55, alpha=0.85, zorder=5,
                   edgecolors='white', linewidth=0.4)
        ax.text(i+1, -0.07, f'n={len(gd)}\nmu={gd.mean():.2f}',
                ha='center', fontsize=9, color=SUB,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(pos16)
    ax.set_xticklabels([o for _, o in valid16], fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
    ax.set_title(f'Openness vs {ylabel.split()[0]}',
                 fontsize=12, fontweight='bold', pad=10, color=TEXT)
    valid_g = [g for g, _ in valid16]
    if len(valid_g) >= 2:
        H16, p16 = stats.kruskal(*valid_g)
        sig16 = '***' if p16<0.001 else '**' if p16<0.01 else '*' if p16<0.05 else 'ns'
        ax.text(0.97, 0.97, f'H={H16:.2f},  p={p16:.3f}  {sig16}',
                transform=ax.transAxes, fontsize=9, color=SUB,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=GRID, alpha=0.9))
fig.suptitle('Plot 16 · Openness (collapsed) vs Key Outcomes',
             fontsize=13, fontweight='bold', color=TEXT, y=1.01)
fig.tight_layout()
path16 = os.path.join(OUTPUT_DIR, '16_openness_vs_outcomes.png')
fig.savefig(path16, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'  saved  {path16}')


# ── Plot 17 · Valence x Openness Interaction Heatmap ────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=BG)
row_cats = ['Open', 'Neutral', 'Closed']
col_cats = ['Positive', 'Negative']
for ax, (metric, label, cmap) in zip(axes, [
        ('Groupthink',   'Groupthink',   'plasma'),
        ('AI_Influence', 'AI Influence', 'cool')]):
    ax.set_facecolor(CARD)
    mat = np.full((len(row_cats), len(col_cats)), np.nan)
    ns  = np.zeros((len(row_cats), len(col_cats)), dtype=int)
    for ri, op in enumerate(row_cats):
        for ci, va in enumerate(col_cats):
            sub = dt[(dt['Openness']==op) & (dt['AIStance']==va)][metric].dropna()
            if len(sub) > 0:
                mat[ri, ci] = sub.mean()
                ns[ri, ci]  = len(sub)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(col_cats, fontsize=11, color=TEXT)
    ax.set_yticks(range(len(row_cats))); ax.set_yticklabels(row_cats, fontsize=11, color=TEXT)
    ax.tick_params(colors=TEXT)
    for ri in range(len(row_cats)):
        for ci in range(len(col_cats)):
            v = mat[ri, ci]; n = ns[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f'{v:.2f}\n(n={n})', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if v > 0.55 else TEXT)
    ax.set_title(f'Mean {label}', fontsize=12, fontweight='bold', color=TEXT, pad=8)
    ax.set_xlabel('Valence', fontsize=11, color=TEXT)
    ax.set_ylabel('Openness', fontsize=11, color=TEXT)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=SUB, labelsize=9)
fig.suptitle('Plot 17 · Valence x Openness Interaction\n'
             '(mean per cell — which framing combo drives groupthink most?)',
             fontsize=12, fontweight='bold', color=TEXT, y=1.04)
fig.tight_layout()
path17 = os.path.join(OUTPUT_DIR, '17_valence_x_openness_interaction.png')
fig.savefig(path17, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close(fig)
print(f'  saved  {path17}')


# ── Plot 18 · Framing -> Dissent Suppressed ─────────────────────
fig, ax = base_fig(w=8, h=5.5)
ax.set_title('Plot 18 · Condition vs Dissent Suppressed  (mean ± 95 % CI)\n'
             '(does negative framing silence dissenters?)',
             fontsize=12, fontweight='bold', pad=12, color=TEXT)
for yi, cond in enumerate(COND_ORDER):
    sub = dt[dt['FullCondition'] == cond]['N_Dissent_Sup'].dropna().values
    if len(sub) == 0:
        continue
    mean = sub.mean()
    ci   = (stats.t.ppf(0.975, df=len(sub)-1) * stats.sem(sub)
            if len(sub) > 1 else 0)
    col  = COND_PAL[cond]
    ax.scatter(sub + rng.normal(0, 0.06, len(sub)),
               np.full(len(sub), yi), color=col, s=45,
               alpha=0.55, zorder=4, edgecolors='none')
    ax.errorbar(mean, yi, xerr=ci, fmt='o', color=col,
                markersize=9, capsize=5, capthick=2, linewidth=2, zorder=6,
                ecolor=col)
    ax.text(1.02, yi, f'n={len(sub)}  mu={mean:.2f}',
            va='center', fontsize=8.5, color=SUB,
            transform=ax.get_yaxis_transform())
ax.set_yticks(list(range(len(COND_ORDER))))
ax.set_yticklabels(COND_ORDER, fontsize=10)
ax.set_xlabel('Mean No. of Decisions with Dissent Suppressed', fontsize=11)
ax.set_xlim(-0.3, 5.5)
pos_ds = dt[dt['AIStance']=='Positive']['N_Dissent_Sup'].dropna().values
neg_ds = dt[dt['AIStance']=='Negative']['N_Dissent_Sup'].dropna().values
t18, p18 = stats.ttest_ind(pos_ds, neg_ds)
sig18 = '***' if p18<0.001 else '**' if p18<0.01 else '*' if p18<0.05 else 'ns'
ax.text(0.98, 0.05,
        f'Positive vs Negative: t={t18:.2f}, p={p18:.3f}  {sig18}\n'
        f'pos mu={pos_ds.mean():.2f}  neg mu={neg_ds.mean():.2f}',
        transform=ax.transAxes, fontsize=9, color=SUB,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '18_condition_vs_dissent_suppressed')


# ── Plot 19 · Framing -> Decision Momentum ──────────────────────
fig, ax = base_fig(w=8, h=5.5)
ax.set_title('Plot 19 · Condition vs Decision Momentum  (mean ± 95 % CI)\n'
             '(early consecutive AI agreement — does framing prime conformity faster?)',
             fontsize=11, fontweight='bold', pad=12, color=TEXT)
for yi, cond in enumerate(COND_ORDER):
    sub = dt[dt['FullCondition'] == cond]['Consec_Agree'].dropna().values
    if len(sub) == 0:
        continue
    mean = sub.mean()
    ci   = (stats.t.ppf(0.975, df=len(sub)-1) * stats.sem(sub)
            if len(sub) > 1 else 0)
    col  = COND_PAL[cond]
    ax.scatter(sub + rng.normal(0, 0.08, len(sub)),
               np.full(len(sub), yi), color=col, s=45,
               alpha=0.55, zorder=4, edgecolors='none')
    ax.errorbar(mean, yi, xerr=ci, fmt='o', color=col,
                markersize=9, capsize=5, capthick=2, linewidth=2, zorder=6,
                ecolor=col)
    ax.text(1.02, yi, f'n={len(sub)}  mu={mean:.2f}',
            va='center', fontsize=8.5, color=SUB,
            transform=ax.get_yaxis_transform())
ax.set_yticks(list(range(len(COND_ORDER))))
ax.set_yticklabels(COND_ORDER, fontsize=10)
ax.set_xlabel('Mean Consecutive AI Agreements at Session Start', fontsize=11)
ax.set_xlim(-0.3, 13.5)
groups_kw19 = [dt[dt['FullCondition']==c]['Consec_Agree'].dropna().values
               for c in COND_ORDER if len(dt[dt['FullCondition']==c]['Consec_Agree'].dropna()) >= 2]
if len(groups_kw19) >= 2:
    H19, p19 = stats.kruskal(*groups_kw19)
    sig19 = '***' if p19<0.001 else '**' if p19<0.01 else '*' if p19<0.05 else 'ns'
    ax.text(0.98, 0.98, f'Kruskal-Wallis  H={H19:.2f},  p={p19:.3f}  {sig19}',
            transform=ax.transAxes, fontsize=9, color=SUB,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=GRID, alpha=0.9))
save_fig(fig, '19_condition_vs_momentum')

print(f"\nDone — all 19 plots saved to '{OUTPUT_DIR}/' (11 original + 3 condition + 5 new framing)")