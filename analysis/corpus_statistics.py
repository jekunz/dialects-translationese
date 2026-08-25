"""Table 1: Corpus statistics for 24 translationese source languages."""

import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'corpus_analysis_results_nopruning.json'), 'r') as f:
    corpus_data = json.load(f)

corpus_100mb = [item for item in corpus_data if item['size'] == '100mb']
df = pd.DataFrame(corpus_100mb)
df['bigram_ttr'] = df['unique_bigrams'] / df['total_bigrams'] * 100

lang_code_map = {
    'swe': 'Swedish', 'isl': 'Icelandic', 'fao': 'Faroese', 'ukr': 'Ukrainian',
    'ces': 'Czech', 'hsb': 'Upper Sorbian', 'fas': 'Persian', 'urd': 'Urdu',
    'kmr': 'Northern Kurdish', 'hun': 'Hungarian', 'ekk': 'Estonian', 'sme': 'Northern Sami',
    'arb': 'Standard Arabic', 'heb': 'Hebrew', 'mlt': 'Maltese', 'ind': 'Indonesian',
    'zsm': 'Standard Malay', 'jav': 'Javanese', 'swh': 'Swahili', 'kin': 'Kinyarwanda',
    'zul': 'Zulu', 'tam': 'Tamil', 'tel': 'Telugu', 'kan': 'Kannada', 'eng': 'English'
}

full_corpus_words = {
    'swe': 35745, 'isl': 1696, 'fao': 101, 'ukr': 25586, 'ces': 35479,
    'hsb': 15, 'fas': 39706, 'urd': 2733, 'kmr': 221, 'hun': 30919,
    'ekk': 6564, 'sme': 25, 'arb': 32813, 'heb': 8463, 'mlt': 287,
    'ind': 60264, 'zsm': 5648, 'jav': 140, 'swh': 570, 'kin': 128,
    'zul': 71.7, 'tam': 1937, 'tel': 891, 'kan': 748, 'eng': 60000
}

lang_stats = df.groupby('language').agg({
    'total_tokens': 'sum',
    'vocab_size': 'sum',
    'type_token_ratio': 'mean',
    'bigram_ttr': 'mean',
}).reset_index()

lang_stats['language_full'] = lang_stats['language'].map(lang_code_map)
lang_stats['full_corpus_M_words'] = lang_stats['language'].map(full_corpus_words)
lang_stats = lang_stats.sort_values('language')

csv_path = os.path.join(OUTPUT_DIR, 'table1_corpus_statistics.csv')
lang_stats.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
