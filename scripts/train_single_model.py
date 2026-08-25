"""Train a GPT-2 model from a JSON config file."""

import json
import sys
import shutil
import sentencepiece as spm
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import torch


def train_model(config_path):
    with open(config_path) as f:
        config = json.load(f)

    model_config = GPT2Config.from_json_file(f"training_code/{config['model_config']}")

    sp = spm.SentencePieceProcessor()
    sp.Load(config['tokenizer_path'])
    model_config.vocab_size = sp.GetPieceSize()

    model = GPT2LMHeadModel(model_config)
    dataset = load_from_disk(config['data_path'])
    split = dataset.train_test_split(test_size=0.1, seed=42)

    def collate_fn(features):
        input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = [torch.tensor(f['attention_mask'], dtype=torch.long) for f in features]
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = input_ids_padded.clone()
        return {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded, 'labels': labels}

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        fp16=True,
        seed=43,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(config['output_dir'])

    shutil.copy(config['tokenizer_path'], f"{config['output_dir']}/tokenizer.model")
    with open(f"{config['output_dir']}/tokenizer_config.json", 'w') as f:
        json.dump({'vocab_size': sp.GetPieceSize()}, f)

    print(f"Saved: {config['output_dir']}")


if __name__ == '__main__':
    train_model(sys.argv[1])
