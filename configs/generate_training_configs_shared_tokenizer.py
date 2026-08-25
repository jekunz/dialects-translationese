import json
import os

LANGUAGES = [
    'hun'
]
SIZES = {
    '5mb': {
        'config': 'gpt_small_config.json',
        'batch_size': 4,
        'num_train_epochs': 10,
    },
    '10mb': {
        'config': 'gpt_small_config.json',
        'batch_size': 8,
        'num_train_epochs': 10,
    },
    '100mb': {
        'config': 'gpt_base_config.json',
        'batch_size': 32,
        'num_train_epochs': 10,
    },
    '1000mb': {
        'config': 'gpt_base_config.json',
        'batch_size': 64,
        'gradient_accumulation': 8,
        'per_device_batch': 8,
        'num_train_epochs': 10,
    }
}

# Shared tokenizer for all models
SHARED_TOKENIZER = 'goldfish-models/eng_latn_1000mb/spiece.model'

CONFIG_DIR = 'finetranslations_models/configs'
os.makedirs(CONFIG_DIR, exist_ok=True)

for lang in LANGUAGES:
    for size, params in SIZES.items():
        config = {
            'language': lang,
            'size': size,
            'model_config': params['config'],
            'tokenizer_path': SHARED_TOKENIZER,
            'data_path': f'finetranslations_models/tokenized_data/{lang}_{size}_test',
            'output_dir': f'finetranslations_models/models/{lang}_{size}_test',
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'num_train_epochs': params['num_train_epochs'],
            'batch_size': params.get('per_device_batch', params['batch_size']),
            'gradient_accumulation_steps': params.get('gradient_accumulation', 1),
        }

        config_file = f"{CONFIG_DIR}/{lang}_{size}_test.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Created config: {config_file}")
