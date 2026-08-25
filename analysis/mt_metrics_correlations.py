"""Table 8: Correlation between MT quality metrics (BLEU, chrF2) and model evaluations.

Predictors: BLEU and chrF2 scores measuring translation quality into each language.
Outcomes:   BLiMP accuracy (100MB models) and FLORES/FineWeb perplexity (100MB + 1000MB).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("colorblind")

mt_data = {
    'lang_code': ['ces', 'ekk', 'fao', 'heb', 'hrv', 'hun', 'ind', 'isl', 'jav',
                  'kan', 'kin', 'kmr', 'mlt', 'fas', 'sme', 'swe', 'swh', 'tam',
                  'tel', 'ukr', 'urd', 'zsm', 'zul'],
    'bleu':  [48.61, 49.20, 33.01, 46.91, 52.08, 47.18, 40.28, 45.16, 39.26,
              24.25, 22.89, 36.37, 54.54, 37.06,  9.15, 55.98, 44.34, 24.16,
              27.46, 45.45, 46.32, 41.53, 38.95],
    'chrf2': [67.41, 67.91, 52.75, 67.25, 70.01, 66.30, 60.51, 63.52, 61.51,
              47.18, 45.84, 57.34, 71.45, 59.18, 29.02, 72.22, 64.76, 46.53,
              50.16, 63.70, 64.72, 62.26, 58.86],
}
df_mt = pd.DataFrame(mt_data)

lang_code_map = {
    'swe': 'Swedish', 'isl': 'Icelandic', 'fao': 'Faroese', 'ukr': 'Ukrainian',
    'ces': 'Czech', 'hsb': 'Upper Sorbian', 'hrv': 'Croatian', 'fas': 'Persian',
    'urd': 'Urdu', 'kmr': 'Northern Kurdish', 'hun': 'Hungarian', 'ekk': 'Estonian',
    'sme': 'Northern Sami', 'arb': 'Standard Arabic', 'heb': 'Hebrew', 'mlt': 'Maltese',
    'ind': 'Indonesian', 'zsm': 'Standard Malay', 'jav': 'Javanese', 'swh': 'Swahili',
    'kin': 'Kinyarwanda', 'zul': 'Zulu', 'tam': 'Tamil', 'tel': 'Telugu', 'kan': 'Kannada',
}
family_colors = {
    'Germanic': ['Swedish', 'Icelandic', 'Faroese'],
    'Slavic': ['Ukrainian', 'Czech', 'Upper Sorbian', 'Croatian'],
    'Iranian': ['Persian', 'Northern Kurdish'],
    'Indo-Aryan': ['Urdu'],
    'Uralic': ['Hungarian', 'Estonian', 'Northern Sami'],
    'Semitic': ['Standard Arabic', 'Hebrew', 'Maltese'],
    'Austronesian': ['Indonesian', 'Standard Malay', 'Javanese'],
    'Niger-Congo': ['Swahili', 'Kinyarwanda', 'Zulu'],
    'Dravidian': ['Tamil', 'Telugu', 'Kannada'],
}
family_map = {lang: fam for fam, langs in family_colors.items() for lang in langs}
df_mt['language'] = df_mt['lang_code'].map(lang_code_map)
df_mt['family'] = df_mt['language'].map(family_map)

colors_palette = sns.color_palette("colorblind", len(family_colors))
family_to_color = {fam: colors_palette[i] for i, fam in enumerate(family_colors)}

CATEGORY_PARADIGMS = {
    'Anaphor Agreement': [
        'anaphor_gender_agreement', 'anaphor_number_agreement',
    ],
    'Irregular Forms': [
        'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
        'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
    ],
    'Determiner-Noun Agreement': [
        'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
        'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2',
        'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1',
        'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1',
    ],
    'Subject-Verb Agreement': [
        'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause',
        'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2',
    ],
    'Argument Structure': [
        'animate_subject_passive', 'animate_subject_trans', 'causative', 'drop_argument',
        'inchoative', 'intransitive', 'passive_1', 'passive_2', 'transitive',
    ],
    'Binding': [
        'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2',
        'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3',
        'principle_A_reconstruction',
    ],
    'Ellipsis': ['ellipsis_n_bar_1', 'ellipsis_n_bar_2'],
    'Control/Raising': [
        'existential_there_object_raising', 'existential_there_subject_raising',
        'expletive_it_object_raising', 'tough_vs_raising_1', 'tough_vs_raising_2',
    ],
    'Quantifiers': [
        'existential_there_quantifiers_1', 'existential_there_quantifiers_2',
        'superlative_quantifiers_1', 'superlative_quantifiers_2',
    ],
    'Filler-Gap': [
        'wh_questions_object_gap', 'wh_questions_subject_gap',
        'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
        'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap',
        'wh_vs_that_with_gap_long_distance',
    ],
    'NPI Licensing': [
        'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2',
        'only_npi_licensor_present', 'only_npi_scope',
        'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope',
    ],
    'Island Effects': [
        'adjunct_island', 'complex_NP_island',
        'coordinate_structure_constraint_complex_left_branch',
        'coordinate_structure_constraint_object_extraction',
        'left_branch_island_echo_question', 'left_branch_island_simple_question',
        'sentential_subject_island', 'wh_island',
    ],
}

blimp_categories = ['Overall'] + list(CATEGORY_PARADIGMS.keys())
mt_predictors = [('bleu', 'BLEU'), ('chrf2', 'chrF2')]


def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


# ── BLiMP correlations (100MB and 1000MB) ─────────────────────────────────────

blimp_model_configs = [
    ('100MB', 'blimp_evaluation_results_100mb.json'),
    ('1000MB', 'blimp_evaluation_results_1000mb.json'),
]

all_blimp_results = []
merged_blimp_by_size = {}

for model_label, blimp_file in blimp_model_configs:
    with open(os.path.join(OUTPUT_DIR, blimp_file), 'r') as f:
        blimp_raw = json.load(f)

    blimp_rows = []
    for item in blimp_raw:
        if item['language'] in ('eng', 'eng_fineweb'):
            continue
        row = {'lang_code': item['language'], 'Overall': item.get('overall_accuracy')}
        by_p = item.get('by_paradigm', {})
        for cat, paradigms in CATEGORY_PARADIGMS.items():
            correct = sum(by_p[p]['correct'] for p in paradigms if p in by_p)
            total = sum(by_p[p]['total'] for p in paradigms if p in by_p)
            row[cat] = (correct / total * 100) if total > 0 else np.nan
        blimp_rows.append(row)
    df_blimp = pd.DataFrame(blimp_rows)

    merged_blimp = pd.merge(df_mt, df_blimp, on='lang_code', how='inner')
    merged_blimp_by_size[model_label] = merged_blimp

    for pred_col, pred_name in mt_predictors:
        for category in blimp_categories:
            if category not in merged_blimp.columns:
                continue
            valid = merged_blimp[[pred_col, category]].dropna()
            if len(valid) < 3:
                continue
            r, p = stats.pearsonr(valid[pred_col], valid[category])
            rho, sp = stats.spearmanr(valid[pred_col], valid[category])
            all_blimp_results.append({
                'Model Size': model_label,
                'Predictor': pred_name,
                'Category': category,
                'n': len(valid),
                'Pearson r': round(r, 3),
                'Pearson p': round(p, 4),
                'Pearson sig': sig(p),
                'Spearman rho': round(rho, 3),
                'Spearman p': round(sp, 4),
                'Spearman sig': sig(sp),
            })

blimp_results_df = pd.DataFrame(all_blimp_results)
blimp_csv = os.path.join(OUTPUT_DIR, 'table8_mt_blimp_correlation.csv')
blimp_results_df.to_csv(blimp_csv, index=False)

# Scatter: each MT metric × Overall BLiMP, one row per model size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for row_idx, model_label in enumerate(['100MB', '1000MB']):
    merged_blimp = merged_blimp_by_size[model_label]
    for col_idx, (pred_col, pred_name) in enumerate(mt_predictors):
        ax = axes[row_idx][col_idx]
        valid_mask = merged_blimp[[pred_col, 'Overall']].notna().all(axis=1)
        for family in family_colors:
            mask = (merged_blimp['family'] == family) & valid_mask
            if mask.sum() > 0:
                ax.scatter(merged_blimp.loc[mask, pred_col], merged_blimp.loc[mask, 'Overall'],
                           alpha=0.8, s=100, color=family_to_color[family], label=family,
                           edgecolors='white', linewidth=0.5)
        for _, row in merged_blimp[valid_mask].iterrows():
            ax.annotate(row['language'], (row[pred_col], row['Overall']),
                        textcoords='offset points', xytext=(5, 3), fontsize=7, alpha=0.8)
        x = merged_blimp.loc[valid_mask, pred_col]
        y = merged_blimp.loc[valid_mask, 'Overall']
        if len(x) >= 3:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), 'k--', alpha=0.5, linewidth=2)
            r_val, p_val = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r={r_val:.3f}\np={p_val:.4f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        ax.set_xlabel(pred_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('BLiMP Overall Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{pred_name} vs BLiMP Overall ({model_label})', fontsize=12)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mt_vs_blimp_overall.pdf'), dpi=300, bbox_inches='tight')

# ── Perplexity correlations (100MB and 1000MB) ────────────────────────────────

with open(os.path.join(OUTPUT_DIR, 'eng_fineweb_evaluation_results_10mb.json'), 'r') as f:
    fineweb_raw = json.load(f)
fineweb_by_size = {
    size: {item['model_trained_on']: item['perplexity']
           for item in fineweb_raw if item['model_size'] == size}
    for size in ('100mb', '1000mb')
}

perp_metrics = [
    ('wikibooks', 'FLORES: Wikibooks'),
    ('wikivoyage', 'FLORES: Wikivoyage'),
    ('wikinews', 'FLORES: Wikinews'),
    ('flores_avg', 'FLORES: Average'),
    ('eng_fineweb', 'English FineWeb'),
]
model_configs = [
    ('100MB', 'flores_perplexity_results_100mb.json', '100mb'),
    ('1000MB', 'flores_perplexity_results_1000mb.json', '1000mb'),
]

all_perp_results = []
merged_perp_by_size = {}

for model_label, flores_file, fw_size in model_configs:
    with open(os.path.join(OUTPUT_DIR, flores_file), 'r') as f:
        flores_raw = json.load(f)

    flores_rows = []
    for item in flores_raw:
        if item['language'] in ('eng', 'eng_fineweb'):
            continue
        flores_rows.append({
            'lang_code': item['language'],
            'wikibooks': item['by_domain']['wikibooks']['perplexity'],
            'wikivoyage': item['by_domain']['wikivoyage']['perplexity'],
            'wikinews': item['by_domain']['wikinews']['perplexity'],
            'flores_avg': item['weighted_average_perplexity'],
        })
    df_flores = pd.DataFrame(flores_rows)
    df_flores['eng_fineweb'] = df_flores['lang_code'].map(fineweb_by_size[fw_size])

    merged_perp = pd.merge(df_mt, df_flores, on='lang_code', how='inner')
    merged_perp_by_size[model_label] = merged_perp

    for pred_col, pred_name in mt_predictors:
        for col, label in perp_metrics:
            if col not in merged_perp.columns:
                continue
            valid = merged_perp[[pred_col, col]].dropna()
            if len(valid) < 3:
                continue
            r, p = stats.pearsonr(valid[pred_col], valid[col])
            rho, sp = stats.spearmanr(valid[pred_col], valid[col])
            all_perp_results.append({
                'Model Size': model_label,
                'Predictor': pred_name,
                'Perplexity Metric': label,
                'n': len(valid),
                'Pearson r': round(r, 3),
                'Pearson p': round(p, 4),
                'Pearson sig': sig(p),
                'Spearman rho': round(rho, 3),
                'Spearman p': round(sp, 4),
                'Spearman sig': sig(sp),
            })

perp_results_df = pd.DataFrame(all_perp_results)
perp_csv = os.path.join(OUTPUT_DIR, 'table9_mt_perplexity_correlation.csv')
perp_results_df.to_csv(perp_csv, index=False)

# Scatter grid: MT metric × FLORES metric, one row per model size
fig, axes = plt.subplots(len(model_configs) * len(mt_predictors), len(perp_metrics),
                         figsize=(len(perp_metrics) * 4, len(model_configs) * len(mt_predictors) * 4))

row_idx = 0
for model_label, _, _ in model_configs:
    merged_perp = merged_perp_by_size[model_label]
    for pred_col, pred_name in mt_predictors:
        for col_idx, (col, label) in enumerate(perp_metrics):
            ax = axes[row_idx][col_idx]
            if col not in merged_perp.columns:
                ax.set_visible(False)
                continue
            valid_mask = merged_perp[[pred_col, col]].notna().all(axis=1)
            if valid_mask.sum() < 3:
                ax.set_visible(False)
                continue
            for family in family_colors:
                mask = (merged_perp['family'] == family) & valid_mask
                if mask.sum() > 0:
                    ax.scatter(merged_perp.loc[mask, pred_col], merged_perp.loc[mask, col],
                               alpha=0.7, s=60, color=family_to_color[family],
                               label=family if (row_idx == 0 and col_idx == len(perp_metrics) - 1) else "",
                               edgecolors='white', linewidth=0.5)
            x = merged_perp.loc[valid_mask, pred_col]
            y = merged_perp.loc[valid_mask, col]
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), 'k--', alpha=0.5, linewidth=1.5)
            r_val, p_val = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r={r_val:.3f}\np={p_val:.4f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
            ax.set_xlabel(pred_name, fontsize=9, fontweight='bold')
            ax.set_ylabel(label.split(': ')[-1] + ' PPL', fontsize=9, fontweight='bold')
            ax.set_title(f'{label} ({model_label})', fontsize=9)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == len(perp_metrics) - 1:
                ax.legend(fontsize=6, loc='best', framealpha=0.9)
        row_idx += 1

plt.suptitle('MT Quality (BLEU / chrF2) vs Evaluation Perplexity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mt_vs_perplexity_metrics.pdf'), dpi=300, bbox_inches='tight')

# ── Filtered 100MB correlations (excluding fao, hsb, kan, sme, zul) ──────────

EXCLUDE_LANGS_100MB = {'fao', 'hsb', 'kan', 'sme', 'zul'}

merged_blimp_100mb_filtered = merged_blimp_by_size['100MB'][
    ~merged_blimp_by_size['100MB']['lang_code'].isin(EXCLUDE_LANGS_100MB)]

blimp_filtered_results = []
for pred_col, pred_name in mt_predictors:
    valid = merged_blimp_100mb_filtered[[pred_col, 'Overall']].dropna()
    if len(valid) < 3:
        continue
    r, p = stats.pearsonr(valid[pred_col], valid['Overall'])
    rho, sp = stats.spearmanr(valid[pred_col], valid['Overall'])
    blimp_filtered_results.append({
        'Model Size': '100MB_filtered',
        'Predictor': pred_name,
        'Category': 'Overall',
        'n': len(valid),
        'Pearson r': round(r, 3),
        'Pearson p': round(p, 4),
        'Pearson sig': sig(p),
        'Spearman rho': round(rho, 3),
        'Spearman p': round(sp, 4),
        'Spearman sig': sig(sp),
    })

blimp_filtered_df = pd.DataFrame(blimp_filtered_results)
blimp_filtered_csv = os.path.join(OUTPUT_DIR, 'table8_mt_blimp_correlation_filtered.csv')
blimp_filtered_df.to_csv(blimp_filtered_csv, index=False)

merged_perp_100mb_filtered = merged_perp_by_size['100MB'][
    ~merged_perp_by_size['100MB']['lang_code'].isin(EXCLUDE_LANGS_100MB)]

perp_filtered_results = []
for pred_col, pred_name in mt_predictors:
    for col, label in perp_metrics:
        if col not in merged_perp_100mb_filtered.columns:
            continue
        valid = merged_perp_100mb_filtered[[pred_col, col]].dropna()
        if len(valid) < 3:
            continue
        r, p = stats.pearsonr(valid[pred_col], valid[col])
        rho, sp = stats.spearmanr(valid[pred_col], valid[col])
        perp_filtered_results.append({
            'Model Size': '100MB_filtered',
            'Predictor': pred_name,
            'Perplexity Metric': label,
            'n': len(valid),
            'Pearson r': round(r, 3),
            'Pearson p': round(p, 4),
            'Pearson sig': sig(p),
            'Spearman rho': round(rho, 3),
            'Spearman p': round(sp, 4),
            'Spearman sig': sig(sp),
        })

perp_filtered_df = pd.DataFrame(perp_filtered_results)
perp_filtered_csv = os.path.join(OUTPUT_DIR, 'table9_mt_perplexity_correlation_filtered.csv')
perp_filtered_df.to_csv(perp_filtered_csv, index=False)