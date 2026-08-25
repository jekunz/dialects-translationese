"""NER statistics over a folder of English .txt files.

Per file:
  - token count, entity count, entity density (per 1k tokens)
  - NER token ratio
  - entity length: mean, std, median, min, max

Per category (across all files):
  - count, density, length mean/std/min/max, top-5 most frequent entities

Usage: python ner_stats.py <folder>
"""

import sys
import os
import statistics
from collections import defaultdict, Counter
import spacy


def fmt_lengths(lengths):
    if not lengths:
        return "n/a"
    mean = statistics.mean(lengths)
    std = statistics.stdev(lengths) if len(lengths) > 1 else 0
    med = statistics.median(lengths)
    return f"mean={mean:.1f}  std={std:.1f}  med={med:.1f}  min={min(lengths)}  max={max(lengths)}"


def main(folder):
    nlp = spacy.load("en_core_web_lg")

    files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    if not files:
        print(f"No .txt files found in {folder}")
        return

    agg_tokens = 0
    agg_ner_tokens = 0
    agg_lengths = []
    agg_by_label = defaultdict(list)   
    agg_text_by_label = defaultdict(Counter)  

    file_summaries = []

    for fname in files:
        text = open(os.path.join(folder, fname), encoding="utf-8").read()
        doc = nlp(text)

        n_tokens = len([t for t in doc if not t.is_space])
        ents = doc.ents

        lengths = [len(ent.text) for ent in ents]
        ner_tokens = sum(len(ent) for ent in ents)
        by_label = defaultdict(list)
        for ent in ents:
            by_label[ent.label_].append(len(ent.text))
            agg_by_label[ent.label_].append(len(ent.text))
            agg_text_by_label[ent.label_][ent.text] += 1

        ratio = ner_tokens / n_tokens if n_tokens else 0
        density = len(ents) / n_tokens * 1000 if n_tokens else 0

        print(f"{'─'*60}")
        print(f"  {fname}")
        print(f"  tokens:        {n_tokens}")
        print(f"  entities:      {len(ents)}  ({density:.1f} per 1k tokens)")
        print(f"  NER tok ratio: {ratio:.3f}  ({ner_tokens}/{n_tokens})")
        print(f"  lengths:       {fmt_lengths(lengths)}")
        if by_label:
            print(f"  by category:")
            for label in sorted(by_label):
                ll = by_label[label]
                print(f"    {label:<16} n={len(ll):<4}  {fmt_lengths(ll)}")

        file_summaries.append({
            'file': fname,
            'tokens': n_tokens,
            'entities': len(ents),
            'density': density,
            'ratio': ratio,
            'lengths': lengths,
        })

        agg_tokens += n_tokens
        agg_ner_tokens += ner_tokens
        agg_lengths.extend(lengths)

    print(f"\n{'═'*60}")
    print("  CROSS-FILE COMPARISON  (sorted by avg entity length)")
    print(f"  {'file':<35} {'ents':>5}  {'density':>7}  {'ratio':>6}  {'avg_len':>7}  {'std_len':>7}  {'max_len':>7}")
    print(f"  {'─'*35}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for s in sorted(file_summaries, key=lambda x: statistics.mean(x['lengths']) if x['lengths'] else 0, reverse=True):
        ll = s['lengths']
        avg = statistics.mean(ll) if ll else 0
        std = statistics.stdev(ll) if len(ll) > 1 else 0
        mx = max(ll) if ll else 0
        print(f"  {s['file']:<35} {s['entities']:>5}  {s['density']:>7.1f}  {s['ratio']:>6.3f}  {avg:>7.1f}  {std:>7.1f}  {mx:>7}")

    print(f"\n{'═'*60}")
    print("  AGGREGATE")
    print(f"  files:         {len(files)}")
    print(f"  tokens:        {agg_tokens}")
    print(f"  entities:      {len(agg_lengths)}  ({len(agg_lengths)/agg_tokens*1000:.1f} per 1k tokens)")
    print(f"  NER tok ratio: {agg_ner_tokens/agg_tokens:.3f}")
    print(f"  lengths:       {fmt_lengths(agg_lengths)}")

    print(f"\n  BY CATEGORY  (sorted by count)")
    print(f"  {'label':<16} {'count':>6}  {'density/1k':>10}  lengths")
    print(f"  {'─'*16}  {'─'*6}  {'─'*10}  {'─'*45}")
    for label, ll in sorted(agg_by_label.items(), key=lambda x: len(x[1]), reverse=True):
        density = len(ll) / agg_tokens * 1000
        top5 = ", ".join(e for e, _ in agg_text_by_label[label].most_common(5))
        print(f"  {label:<16} {len(ll):>6}  {density:>10.2f}  {fmt_lengths(ll)}")
        print(f"  {'':16}  top: {top5}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ner_stats.py <folder>")
        sys.exit(1)
    main(sys.argv[1])
