"""Tables 4-5: BLiMP accuracy and correlation analysis."""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'corpus_analysis_results_nopruning.json'), 'r') as f:
    corpus_data = json.load(f)

corpus_100mb = [item for item in corpus_data if item['size'] == '100mb']
df_100mb = pd.DataFrame(corpus_100mb)
df_100mb['bigram_ttr'] = df_100mb['unique_bigrams'] / df_100mb['total_bigrams'] * 100

lang_stats_100mb = df_100mb.groupby('language').agg({
    'total_tokens': 'sum', 'vocab_size': 'sum',
    'type_token_ratio': 'mean', 'bigram_ttr': 'mean',
}).reset_index()

corpus_table_data = {
    'language': ['Swedish', 'Icelandic', 'Faroese', 'Ukrainian', 'Czech', 'Upper Sorbian',
                 'Persian', 'Urdu', 'Northern Kurdish', 'Hungarian', 'Estonian', 'Northern Sami',
                 'Standard Arabic', 'Hebrew', 'Maltese', 'Indonesian', 'Standard Malay', 'Javanese',
                 'Swahili', 'Kinyarwanda', 'Zulu', 'Tamil', 'Telugu', 'Kannada'],
    'words_millions': [35745, 1696, 101, 25586, 35479, 15, 39706, 2733, 221, 30919, 6564, 25,
                       32813, 8463, 287, 60264, 5648, 140, 570, 128, 71.7, 1937, 891, 748],
}
df_full_corpus = pd.DataFrame(corpus_table_data)
df_full_corpus['words'] = df_full_corpus['words_millions'] * 1_000_000

lang2vec_data = {
    'lang_code': ['arb', 'ces', 'ekk', 'fao', 'fas', 'heb', 'hsb', 'hun', 'ind', 'isl', 'jav',
                  'kan', 'kin', 'kmr', 'mlt', 'sme', 'swe', 'swh', 'tam', 'tel', 'ukr', 'urd',
                  'zsm', 'zul'],
    'sim_to_eng': [0.611, 0.685, np.nan, np.nan, 0.464, 0.724, np.nan, 0.681, 0.724, 0.852, 0.000,
                   0.588, 0.529, 0.489, 0.775, 0.710, 0.920, 0.421, 0.520, 0.500, 0.876, 0.667,
                   np.nan, 0.400]
}
df_lang2vec = pd.DataFrame(lang2vec_data)

lang_code_map = {
    'swe': 'Swedish', 'isl': 'Icelandic', 'fao': 'Faroese', 'ukr': 'Ukrainian',
    'ces': 'Czech', 'hsb': 'Upper Sorbian', 'fas': 'Persian', 'urd': 'Urdu',
    'kmr': 'Northern Kurdish', 'hun': 'Hungarian', 'ekk': 'Estonian', 'sme': 'Northern Sami',
    'arb': 'Standard Arabic', 'heb': 'Hebrew', 'mlt': 'Maltese', 'ind': 'Indonesian',
    'zsm': 'Standard Malay', 'jav': 'Javanese', 'swh': 'Swahili', 'kin': 'Kinyarwanda',
    'zul': 'Zulu', 'tam': 'Tamil', 'tel': 'Telugu', 'kan': 'Kannada', 'eng': 'English'
}
reverse_map = {v: k for k, v in lang_code_map.items()}

CATEGORY_PARADIGMS = {
    'Anaphor Agreement': ['anaphor_gender_agreement', 'anaphor_number_agreement'],
    'Irregular Forms': ['irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
                        'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2'],
    'Determiner-Noun Agreement': ['determiner_noun_agreement_1', 'determiner_noun_agreement_2',
                                  'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2',
                                  'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1',
                                  'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1'],
    'Subject-Verb Agreement': ['distractor_agreement_relational_noun', 'distractor_agreement_relative_clause',
                                'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2'],
    'Argument Structure': ['animate_subject_passive', 'animate_subject_trans', 'causative', 'drop_argument',
                           'inchoative', 'intransitive', 'passive_1', 'passive_2', 'transitive'],
    'Binding': ['principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2',
                'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction'],
    'Ellipsis': ['ellipsis_n_bar_1', 'ellipsis_n_bar_2'],
    'Control/Raising': ['existential_there_object_raising', 'existential_there_subject_raising',
                        'expletive_it_object_raising', 'tough_vs_raising_1', 'tough_vs_raising_2'],
    'Quantifiers': ['existential_there_quantifiers_1', 'existential_there_quantifiers_2',
                    'superlative_quantifiers_1', 'superlative_quantifiers_2'],
    'Filler-Gap': ['wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance',
                   'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap',
                   'wh_vs_that_with_gap_long_distance'],
    'NPI Licensing': ['matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2',
                      'only_npi_licensor_present', 'only_npi_scope',
                      'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope'],
    'Island Effects': ['adjunct_island', 'complex_NP_island',
                       'coordinate_structure_constraint_complex_left_branch',
                       'coordinate_structure_constraint_object_extraction',
                       'left_branch_island_echo_question', 'left_branch_island_simple_question',
                       'sentential_subject_island', 'wh_island'],
}

with open(os.path.join(OUTPUT_DIR, 'blimp_evaluation_results_100mb.json'), 'r') as f:
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
blimp_100mb = pd.DataFrame(blimp_rows)
blimp_100mb['Language'] = blimp_100mb['lang_code'].map(lang_code_map)


def merge_all_data(lang_stats, blimp_data):
    ls = lang_stats.copy()
    ls['language_full'] = ls['language'].map(lang_code_map)
    merged = pd.merge(ls, df_full_corpus, left_on='language_full', right_on='language', how='inner')
    if 'language_y' in merged.columns:
        merged = merged.drop(columns=['language_y']).rename(columns={'language_x': 'language'})
    merged['lang_code'] = merged['language_full'].map(reverse_map)
    merged = pd.merge(merged, df_lang2vec, on='lang_code', how='left')
    merged = pd.merge(merged, blimp_data, on='lang_code', how='inner')
    merged = merged[merged['lang_code'] != 'eng']
    if 'Language_x' in merged.columns:
        merged = merged.drop(columns=['Language_y']).rename(columns={'Language_x': 'Language'})
    return merged


merged_100mb = merge_all_data(lang_stats_100mb, blimp_100mb)

categories = ['Overall', 'Anaphor Agreement', 'Irregular Forms', 'Determiner-Noun Agreement',
              'Subject-Verb Agreement', 'Argument Structure', 'Binding', 'Ellipsis',
              'Control/Raising', 'Quantifiers', 'Filler-Gap', 'NPI Licensing', 'Island Effects']

predictors = [
    ('words', 'Corpus Size (words)'),
    ('type_token_ratio', 'Type-Token Ratio (%)'),
    ('sim_to_eng', 'Lang2vec Similarity to English'),
]


def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


results = []
for predictor_col, predictor_name in predictors:
    for category in categories:
        if category not in merged_100mb.columns:
            continue
        valid = merged_100mb[[predictor_col, category]].dropna()
        if len(valid) < 3:
            continue
        r, p = stats.pearsonr(valid[predictor_col], valid[category])
        rho, sp = stats.spearmanr(valid[predictor_col], valid[category])
        results.append({
            'Model Size': '100MB', 'Predictor': predictor_name,
            'Category': category, 'n': len(valid),
            'Pearson r': r, 'Pearson p': p,
            'Spearman rho': rho, 'Spearman p': sp,
            'Pearson sig': sig(p), 'Spearman sig': sig(sp)
        })

results_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, 'tables4_5_blimp_correlations.csv')
results_df.to_csv(csv_path, index=False)

EXCLUDE_LANGS_100MB = {'fao', 'hsb', 'kan', 'sme', 'zul'}
merged_100mb_filtered = merged_100mb[~merged_100mb['lang_code'].isin(EXCLUDE_LANGS_100MB)]

results_filtered = []
for predictor_col, predictor_name in predictors:
    if 'Overall' not in merged_100mb_filtered.columns:
        continue
    valid = merged_100mb_filtered[[predictor_col, 'Overall']].dropna()
    if len(valid) < 3:
        continue
    r, p = stats.pearsonr(valid[predictor_col], valid['Overall'])
    rho, sp = stats.spearmanr(valid[predictor_col], valid['Overall'])
    results_filtered.append({
        'Model Size': '100MB_filtered', 'Predictor': predictor_name,
        'Category': 'Overall', 'n': len(valid),
        'Pearson r': r, 'Pearson p': p,
        'Spearman rho': rho, 'Spearman p': sp,
        'Pearson sig': sig(p), 'Spearman sig': sig(sp)
    })

results_filtered_df = pd.DataFrame(results_filtered)
csv_filtered_path = os.path.join(OUTPUT_DIR, 'tables4_5_blimp_correlations_filtered.csv')
results_filtered_df.to_csv(csv_filtered_path, index=False)
