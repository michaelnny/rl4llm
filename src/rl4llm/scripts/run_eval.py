"""Runs evaluation on a trained model using zero-shot prompting."""

import argparse
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl4llm.data import load_and_combine_datasets
from rl4llm.utils import create_model_and_tokenizer, get_runtime_device, save_to_jsonl_file, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM using single turn instruction.')
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name and tokenizer to be loaded',
    )
    parser.add_argument(
        '--model-ckpt-dir',
        type=str,
        required=False,
        # default='/home/michael/.llama/checkpoints/Llama3.2-3B-Instruct',
        help='Checkpoint for the model to be loaded',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Evaluation batch size',
    )
    # parser.add_argument(
    #     '--max-samples',
    #     type=int,
    #     default=256,
    #     help='Maximum number of sample to run evaluation',
    # )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM generation temperature',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=1.0,
        help='LLM generation top-p sampling',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=0,
        help='LLM generation top-k sampling',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of generation tokens',
    )
    parser.add_argument(
        '--sample-path',
        type=str,
        help='To save generated path',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=43,
        help='Runtime seed for reproducibility',
    )
    return parser.parse_args()


@torch.inference_mode()
def main():

    args = parse_args()
    seed = args.seed
    set_seed(seed)

    logger = setup_logger()

    device = get_runtime_device()
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)

    logger.info(f"Loading datasets: {args.tasks}")
    _, test_ds = load_and_combine_datasets(args.tasks)

    if args.model_ckpt_dir:
        logger.info(f"Loading model from: {args.model_ckpt_dir}")
        model = AutoModelForCausalLM.from_pretrained(args.model_ckpt_dir)
    else:
        logger.info(f"Loading model from: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    eval_decoding = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_new_tokens': args.max_new_tokens,
    }

    with tqdm(
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() and device.type == 'cuda',
        ),
        desc='Evaluating',
    ) as pbar:
        pass

    # if args.sample_path:
    #     logger.info(f"Saving samples to: {args.sample_path}")
    #     save_to_jsonl_file(samples, args.sample_path)


if __name__ == '__main__':
    main()
