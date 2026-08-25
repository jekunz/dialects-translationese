"""Cross-evaluate perplexity of multilingual models on test datasets."""

import json
import torch
import math
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from datasets import load_from_disk
import sentencepiece as spm
from tqdm import tqdm

LANGUAGES = [
    'arb', 'ces', 'ekk', 'eng', 'eng_fineweb', 'fao', 'fas', 'heb', 'hsb', 'hun',
    'ind', 'isl', 'jav', 'kan', 'kin', 'kmr', 'mlt', 'sme', 'swe', 'swh',
    'tam', 'tel', 'ukr', 'urd', 'zsm', 'zul'
]

DATASET_SIZE = '10mb'
MODEL_SIZES = ['100mb', '1000mb']
EVAL_MODELS = None


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


def calculate_perplexity(model, dataset, device, batch_size=32, max_samples=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            end_idx = min(i + batch_size, num_samples)
            batch = dataset[i:end_idx]

            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']]
            attention_mask_list = [torch.tensor(mask, dtype=torch.long) for mask in batch['attention_mask']]

            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            batch_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def get_dataset_name(lang):
    if lang == 'eng':
        return 'eng_fineweb-edu'
    elif lang == 'eng_fineweb':
        return 'eng_fineweb_fineweb'
    return lang


def get_model_name(lang):
    if lang == 'eng':
        return 'eng_fineweb-edu'
    elif lang == 'eng_fineweb':
        return 'eng_fineweb_fineweb'
    return lang


def find_available_datasets(base_path='finetranslations_models/tokenized_data'):
    available = []
    for lang in LANGUAGES:
        data_path = f'{base_path}/{get_dataset_name(lang)}_{DATASET_SIZE}_test'
        if os.path.exists(data_path):
            available.append(lang)
    return available


def find_available_models(base_path='finetranslations_models/models'):
    available = []
    for model_size in MODEL_SIZES:
        for lang in LANGUAGES:
            model_path = f'{base_path}/{get_model_name(lang)}_{model_size}_test'
            if os.path.exists(model_path):
                available.append((lang, model_size))
    return available


def main():
    results = []
    available_datasets = find_available_datasets()

    global EVAL_MODELS
    if EVAL_MODELS is None:
        EVAL_MODELS = find_available_models()

    for model_lang, model_size in EVAL_MODELS:
        model_path = f'finetranslations_models/models/{get_model_name(model_lang)}_{model_size}_test'
        tokenizer_path = f'goldfish-models/eng_latn_1000mb/spiece.model'

        try:
            model, tokenizer, device = load_model(model_path, tokenizer_path)
        except Exception as e:
            print(f"Error loading model {model_lang}: {e}")
            continue

        for data_lang in available_datasets:
            data_path = f'finetranslations_models/tokenized_data/{get_dataset_name(data_lang)}_{DATASET_SIZE}_test'
            try:
                dataset = load_from_disk(data_path)
                perplexity, loss = calculate_perplexity(model, dataset, device, batch_size=32)
                results.append({
                    'model_trained_on': model_lang, 'model_size': model_size,
                    'evaluated_on': data_lang, 'dataset_size_used': DATASET_SIZE,
                    'perplexity': perplexity, 'loss': loss, 'num_samples': len(dataset)
                })
            except Exception as e:
                print(f"Error evaluating {model_lang} on {data_lang}: {e}")

        del model
        torch.cuda.empty_cache()

    model_sizes_str = '_'.join(MODEL_SIZES)
    output_file = f'finetranslations_models/cross_evaluation_results_data{DATASET_SIZE}_models{model_sizes_str}.json'
    with open(output_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
