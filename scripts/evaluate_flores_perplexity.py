"""Evaluate trained models on English FLORES+ data, separated by domain."""

import json
import torch
import math
import os
from transformers import GPT2LMHeadModel
from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm

LANGUAGES = [
    'arb', 'ces', 'ekk', 'eng', 'eng_fineweb', 'fao', 'fas', 'heb', 'hsb', 'hun',
    'ind', 'isl', 'jav', 'kan', 'kin', 'kmr', 'mlt', 'sme', 'swe', 'swh',
    'tam', 'tel', 'ukr', 'urd', 'zsm', 'zul'
]

SIZE = '1000mb'
EVAL_MODELS = None
DOMAINS = ['wikibooks', 'wikivoyage', 'wikinews']


def load_model(model_path, tokenizer_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    sp = spm.SentencePieceProcessor()
    if not tokenizer_path.endswith('.model'):
        if not os.path.exists(tokenizer_path):
            from huggingface_hub import hf_hub_download
            tokenizer_path = hf_hub_download(repo_id=tokenizer_path, filename='spiece.model')
        else:
            tokenizer_path = f'{tokenizer_path}/spiece.model'
    sp.Load(tokenizer_path)

    return model, sp, device


def calculate_perplexity_on_text(model, sp, texts, device, max_length=512, batch_size=32):
    from torch.nn.utils.rnn import pad_sequence

    model.eval()
    total_loss = 0
    total_tokens = 0

    tokenized_texts = []
    for text in texts:
        token_ids = sp.EncodeAsIds(text)
        if len(token_ids) == 0:
            continue
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        tokenized_texts.append(token_ids)

    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_texts), batch_size), desc="  Processing"):
            batch_token_ids = tokenized_texts[i:i+batch_size]
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in batch_token_ids]
            attention_mask_list = [torch.ones(len(ids), dtype=torch.long) for ids in batch_token_ids]

            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            batch_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens

    if total_tokens == 0:
        return float('inf'), float('inf')

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def get_model_name(lang):
    if lang == 'eng':
        return 'eng_fineweb-edu'
    elif lang == 'eng_fineweb':
        return 'eng_fineweb_fineweb'
    return lang


def find_available_models(base_path='finetranslations_models/models'):
    available = []
    for lang in LANGUAGES:
        model_path = f'{base_path}/{get_model_name(lang)}_{SIZE}_test'
        if os.path.exists(model_path):
            available.append(lang)
    return available


def main():
    all_results = []

    global EVAL_MODELS
    if EVAL_MODELS is None:
        EVAL_MODELS = find_available_models()

    flores_dataset = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split='devtest')

    for lang in EVAL_MODELS:
        model_name = get_model_name(lang)
        model_path = f'finetranslations_models/models/{model_name}_{SIZE}_test'
        tokenizer_path = f'goldfish-models/eng_latn_1000mb'

        try:
            model, sp, device = load_model(model_path, tokenizer_path)
            results_by_domain = {}

            if 'domain' in flores_dataset.column_names:
                for domain in DOMAINS:
                    domain_data = flores_dataset.filter(lambda x: x['domain'] == domain)
                    if len(domain_data) == 0:
                        continue
                    texts = [item['text'] for item in domain_data]
                    perplexity, loss = calculate_perplexity_on_text(model, sp, texts, device)
                    results_by_domain[domain] = {
                        'perplexity': round(perplexity, 2),
                        'loss': round(loss, 4),
                        'num_examples': len(texts)
                    }
            else:
                texts = [item['text'] for item in flores_dataset]
                perplexity, loss = calculate_perplexity_on_text(model, sp, texts, device)
                results_by_domain['full_dataset'] = {
                    'perplexity': round(perplexity, 2),
                    'loss': round(loss, 4),
                    'num_examples': len(texts)
                }

            total_examples = sum(d['num_examples'] for d in results_by_domain.values())
            weighted_ppl = sum(d['perplexity'] * d['num_examples'] for d in results_by_domain.values()) / total_examples if results_by_domain else None

            all_results.append({
                'language': lang, 'size': SIZE, 'model_path': model_path,
                'by_domain': results_by_domain,
                'weighted_average_perplexity': round(weighted_ppl, 2) if weighted_ppl else None,
                'total_examples': total_examples if results_by_domain else 0
            })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({
                'language': lang, 'size': SIZE, 'model_path': model_path,
                'error': str(e), 'by_domain': {}, 'weighted_average_perplexity': None
            })

    output_file = f'finetranslations_models/flores_perplexity_results_{SIZE}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
