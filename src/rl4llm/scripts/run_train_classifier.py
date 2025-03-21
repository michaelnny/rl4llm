"""Script to train a small coherent classifier on a single GPU."""

import argparse
import math
import os
import random
import sys
from traceback import format_exc

import torch

from rl4llm.core.classifier_trainer import ClassifierConfig, ClassifierTrainer
from rl4llm.models import ClassifierModel
from rl4llm.utils import (
    build_classification_model_and_tokenizer,
    build_longformer_classification_model_and_tokenizer,
    create_optimizer_and_scheduler,
    load_from_jsonl_file,
    load_yaml_config_file,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Trains coherent classifier')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/coherent_classifier_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    return parser.parse_args()


def init_head_weights(model: ClassifierModel) -> None:
    """Initialize the weights of the classification head."""

    if hasattr(model, 'classification_head'):
        head_module = model.classification_head

        for m in head_module.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * model.config.num_hidden_layers))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def main():
    """Starts coherent classifier training loop."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script only supports run on GPU with BF16 mode.')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    seed = int(config.get('job').get('seed', 142))
    artifacts_path = config.get('job').get('artifacts_path')
    train_path = config.get('job').get('dataset').get('train_path')
    test_path = config.get('job').get('dataset').get('test_path')
    max_train_samples = config.get('job').get('max_train_samples', None)
    max_test_samples = config.get('job').get('max_test_samples', None)
    set_seed(seed)

    logger = setup_logger()
    classifier_config = ClassifierConfig(**config['config'])

    train_ds, test_ds = load_from_jsonl_file(train_path), load_from_jsonl_file(test_path)

    if max_train_samples is not None and max_train_samples < len(train_ds):
        logger.info(f"Randomly select {max_train_samples} training samples")
        random.shuffle(train_ds)
        train_ds = train_ds[:max_train_samples]
    else:
        logger.info(f'Number of training samples: {len(train_ds)}')

    if max_test_samples is not None and max_test_samples < len(test_ds):
        logger.info(f"Randomly select {max_test_samples} testing samples")
        random.shuffle(test_ds)
        test_ds = test_ds[:max_test_samples]
    else:
        logger.info(f'Number of testing samples: {len(test_ds)}')

    device = torch.device('cuda')
    torch_dtype = torch.bfloat16

    # model, tokenizer = build_classification_model_and_tokenizer(config['model'], torch_dtype)

    # logger.info("Freezing the pretrained model parameters")
    # for param in model.pretrained_model.parameters():
    #     param.requires_grad = False

    # logger.info("Initalizing classification head weights")
    # init_head_weights(model)

    model, tokenizer = build_longformer_classification_model_and_tokenizer(config['model'], torch_dtype)

    logger.info('Freezing the pretrained model parameters')
    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    logger.info('Initalizing classification head weights')
    init_head_weights(model)

    total_updates_per_epoch = math.ceil(
        len(train_ds) / classifier_config.batch_size / classifier_config.gradient_accumulate_steps
    )
    total_steps = classifier_config.num_epochs * total_updates_per_epoch  # Total gradient update steps

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        optimizer_config=config['optimizer'],
        scheduler_config=config.get('scheduler', None),
        total_steps=total_steps,
    )

    trainer = ClassifierTrainer(
        config=classifier_config,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        torch_dtype=torch_dtype,
        artifacts_path=artifacts_path,
        logger=logger,
    )

    def handle_exit():
        trainer.on_exit()

    try:
        trainer.train(log_hyper_params=config)
    except KeyboardInterrupt:
        logger.info('\nKeyboardInterrupt received in main loop. Shutting down...')
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(format_exc())
        sys.exit(1)
    finally:
        handle_exit()
        logger.info('Exiting main program.')


if __name__ == '__main__':
    main()
