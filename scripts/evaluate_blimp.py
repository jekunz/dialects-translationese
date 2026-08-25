"""Evaluate trained models on BLiMP (Benchmark of Linguistic Minimal Pairs)."""

import json
import torch
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

PARADIGMS = [
    'adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
    'animate_subject_passive', 'animate_subject_trans', 'causative',
    'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch',
    'coordinate_structure_constraint_object_extraction',
    'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
    'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2',
    'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1',
    'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1',
    'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause',
    'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
    'existential_there_object_raising', 'existential_there_quantifiers_1',
    'existential_there_quantifiers_2', 'existential_there_subject_raising',
    'expletive_it_object_raising', 'inchoative', 'intransitive',
    'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
    'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
    'left_branch_island_echo_question', 'left_branch_island_simple_question',
    'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2',
    'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2',
    'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2',
    'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3',
    'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2',
    'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope',
    'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2',
    'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive',
    'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
    'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap',
    'wh_vs_that_with_gap_long_distance'
]


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


def calculate_sentence_log_probability(model, sp, sentence, device):
    token_ids = sp.EncodeAsIds(sentence)
    if len(token_ids) == 0:
        return float('-inf')
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        return -outputs.loss.item() * len(token_ids)


def evaluate_blimp(model, sp, device, max_examples=None):
    results_by_paradigm = {}
    total_correct = 0
    total_examples = 0

    for paradigm in tqdm(PARADIGMS, desc="Evaluating paradigms"):
        try:
            dataset = load_dataset("nyu-mll/blimp", paradigm, split='train')
            if max_examples:
                dataset = dataset.select(range(min(max_examples, len(dataset))))

            correct = 0
            for example in tqdm(dataset, desc=f"  {paradigm}", leave=False):
                good_log_prob = calculate_sentence_log_probability(model, sp, example['sentence_good'], device)
                bad_log_prob = calculate_sentence_log_probability(model, sp, example['sentence_bad'], device)
                if good_log_prob > bad_log_prob:
                    correct += 1

            accuracy = (correct / len(dataset) * 100) if len(dataset) > 0 else 0
            results_by_paradigm[paradigm] = {
                'correct': correct, 'total': len(dataset), 'accuracy': round(accuracy, 2)
            }
            total_correct += correct
            total_examples += len(dataset)

        except Exception as e:
            results_by_paradigm[paradigm] = {
                'correct': 0, 'total': 0, 'accuracy': 0.0, 'error': str(e)
            }

    overall_accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0
    return {
        'overall_accuracy': round(overall_accuracy, 2),
        'total_correct': total_correct,
        'total_examples': total_examples,
        'by_paradigm': results_by_paradigm
    }


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

    for lang in EVAL_MODELS:
        model_name = get_model_name(lang)
        model_path = f'finetranslations_models/models/{model_name}_{SIZE}_test'
        tokenizer_path = f'goldfish-models/eng_latn_1000mb/spiece.model'

        try:
            model, sp, device = load_model(model_path, tokenizer_path)
            results = evaluate_blimp(model, sp, device, max_examples=None)
            results['language'] = lang
            results['size'] = SIZE
            results['model_path'] = model_path
            all_results.append(results)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            all_results.append({
                'language': lang, 'size': SIZE, 'model_path': model_path,
                'error': str(e), 'overall_accuracy': None,
                'total_correct': None, 'total_examples': None, 'by_paradigm': {}
            })

        output_file = f'finetranslations_models/blimp_evaluation_results_{SIZE}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
