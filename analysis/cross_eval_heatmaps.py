"""Figure 2: Cross-evaluation perplexity heatmaps for 100MB and 1000MB models."""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'cross_evaluation_results.json'), 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)


def create_heatmap(model_size, df, exclude_languages=None):
    df_filtered = df[df['model_size'] == model_size].copy()
    if exclude_languages:
        df_filtered = df_filtered[
            ~df_filtered['model_trained_on'].isin(exclude_languages) &
            ~df_filtered['evaluated_on'].isin(exclude_languages)
        ]

    pivot = df_filtered.pivot(index='model_trained_on', columns='evaluated_on', values='perplexity')
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(14, 12))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Perplexity'}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Evaluated On (Language)', fontsize=12, fontweight='bold')
    plt.ylabel('Model Trained On (Language)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'figure2_cross_eval_heatmap_{model_size}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'Saved: {output_path}')


excluded_100mb = ['eng', 'eng_fineweb']
create_heatmap('100mb', df, exclude_languages=excluded_100mb)

excluded_1000mb = ['fao', 'hsb', 'kin', 'sme', 'zul', 'eng', 'eng_fineweb']
create_heatmap('1000mb', df, exclude_languages=excluded_1000mb)
