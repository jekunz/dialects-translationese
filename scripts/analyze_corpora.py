"""Analyze pre-training corpora statistics: tokens, vocabulary, TTR, bigrams."""

import json
import os
import re
from collections import Counter
from tqdm import tqdm

LANGUAGES = [
    'arb', 'ces', 'ekk', 'eng', 'eng_fineweb', 'fao', 'fas',
    'heb', 'hsb', 'hun', 'ind', 'isl', 'jav', 'kan', 'kin',
    'kmr', 'mlt', 'sme', 'swe', 'swh', 'tam', 'tel',
    'ukr', 'urd', 'zsm', 'zul'
]
SIZES = ['5mb', '10mb', '100mb', '1000mb']
DATA_DIR = 'finetranslations_models/data'


def analyze_corpus(file_path):
    if not os.path.exists(file_path):
        return None

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    token_counter = Counter()
    bigram_counter = Counter()
    total_tokens = 0
    total_chars = 0
    num_lines = 0
    num_sentences = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
            num_lines += 1
            tokens = line.strip().split()
            token_counter.update(tokens)
            total_tokens += len(tokens)
            total_chars += len(line)

            for i in range(len(tokens) - 1):
                bigram_counter[(tokens[i], tokens[i+1])] += 1

            sentences = [s for s in re.split(r'[.!?]+', line.strip()) if s.strip()]
            num_sentences += len(sentences)

    vocab_size = len(token_counter)
    type_token_ratio = (vocab_size / total_tokens * 100) if total_tokens > 0 else 0

    top_50 = token_counter.most_common(50)
    top_50_with_percentages = [
        {'rank': i + 1, 'token': token, 'count': count,
         'percentage': round((count / total_tokens * 100), 4) if total_tokens > 0 else 0}
        for i, (token, count) in enumerate(top_50)
    ]
    top_50_count = sum(count for _, count in top_50)
    top_50_percentage = (top_50_count / total_tokens * 100) if total_tokens > 0 else 0

    top_100_count = sum(count for _, count in token_counter.most_common(100))
    top_500_count = sum(count for _, count in token_counter.most_common(500))
    top_1000_count = sum(count for _, count in token_counter.most_common(1000))

    avg_sentence_length = round(total_tokens / num_sentences, 2) if num_sentences > 0 else 0

    total_bigrams = sum(bigram_counter.values())
    unique_bigrams = len(bigram_counter)

    top_50_bigrams = bigram_counter.most_common(50)
    top_50_bigrams_with_percentages = [
        {'rank': i + 1, 'bigram': f"{bigram[0]} {bigram[1]}", 'count': count,
         'percentage': round((count / total_bigrams * 100), 4) if total_bigrams > 0 else 0}
        for i, (bigram, count) in enumerate(top_50_bigrams)
    ]
    top_50_bigrams_count = sum(count for _, count in top_50_bigrams)

    def pct(count, total):
        return round((count / total * 100), 2) if total > 0 else 0

    return {
        'file_path': file_path,
        'file_size_bytes': file_size_bytes,
        'file_size_mb': round(file_size_mb, 2),
        'num_lines': num_lines,
        'num_sentences': num_sentences,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'type_token_ratio': round(type_token_ratio, 4),
        'top_50_tokens': top_50_with_percentages,
        'top_50_count': top_50_count,
        'top_50_cumulative_percentage': round(top_50_percentage, 2),
        'top_100_percentage': pct(top_100_count, total_tokens),
        'top_500_percentage': pct(top_500_count, total_tokens),
        'top_1000_percentage': pct(top_1000_count, total_tokens),
        'avg_tokens_per_line': round(total_tokens / num_lines, 2) if num_lines > 0 else 0,
        'avg_sentence_length': avg_sentence_length,
        'total_bigrams': total_bigrams,
        'unique_bigrams': unique_bigrams,
        'top_50_bigrams': top_50_bigrams_with_percentages,
        'top_50_bigrams_count': top_50_bigrams_count,
        'top_50_bigrams_cumulative_percentage': pct(top_50_bigrams_count, total_bigrams),
        'top_100_bigrams_percentage': pct(sum(count for _, count in bigram_counter.most_common(100)), total_bigrams),
        'top_500_bigrams_percentage': pct(sum(count for _, count in bigram_counter.most_common(500)), total_bigrams),
        'top_1000_bigrams_percentage': pct(sum(count for _, count in bigram_counter.most_common(1000)), total_bigrams),
    }


def main():
    all_results = []

    for lang in LANGUAGES:
        for size in SIZES:
            file_path = f"{DATA_DIR}/{lang}_english_{size}_test.txt"
            result = analyze_corpus(file_path)
            if result:
                result['language'] = lang
                result['size'] = size
                all_results.append(result)

    output_file = 'finetranslations_models/corpus_analysis_results_nopruning.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
