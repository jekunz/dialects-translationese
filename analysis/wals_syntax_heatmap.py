"""Figure 1: WALS syntax feature cosine similarity heatmap."""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import lang2vec.lang2vec as l2v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES = sorted([
    'arb', 'ces', 'ekk', 'fao', 'fas', 'heb', 'hsb', 'hun', 'ind', 'isl',
    'jav', 'kan', 'kin', 'kmr', 'mlt', 'sme', 'swe', 'swh', 'tam', 'tel',
    'ukr', 'urd', 'zsm', 'zul'
])

FEATURE_SET = "syntax_wals"

features = l2v.get_features(LANGUAGES, FEATURE_SET)

def to_float_or_nan(val):
    if val == '--' or val == 'N/A' or val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

if isinstance(list(features.values())[0], list):
    X = np.array([[to_float_or_nan(v) for v in features[lang]] for lang in LANGUAGES], dtype=float)
else:
    feature_names = list(features.keys())
    X = np.array([[to_float_or_nan(features[f][lang]) for f in feature_names] for lang in LANGUAGES], dtype=float)

n = len(LANGUAGES)
similarity_matrix = np.full((n, n), np.nan)

for i in range(n):
    for j in range(n):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            if mask.sum() > 0:
                similarity_matrix[i, j] = cosine_similarity(
                    X[i][mask].reshape(1, -1),
                    X[j][mask].reshape(1, -1)
                )[0, 0]

fig, ax = plt.subplots(figsize=(14, 12))
cmap = plt.cm.YlOrRd
cmap.set_bad(color='lightgray')

im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(LANGUAGES, rotation=90, fontsize=16)
ax.set_yticklabels(LANGUAGES, fontsize=16)
ax.grid(False)
ax.set_frame_on(False)

for i in range(n):
    for j in range(n):
        if np.isnan(similarity_matrix[i, j]):
            ax.text(j, i, 'N/A', ha="center", va="center", color="black", fontsize=12)
        else:
            color = "white" if similarity_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', ha="center", va="center",
                    color=color, fontsize=12)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Cosine Similarity', fontsize=16)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'figure1_wals_syntax_heatmap.pdf')
plt.savefig(output_path, format='pdf')
print(f"Saved: {output_path}")
