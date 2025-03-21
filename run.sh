#!/bin/bash

PYTHONPATH=src python3 -m rl4llm.scripts.build_coherent_dataset

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_classifier

# Run the first job and wait for it to complete
# PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/grpo_config.yaml

# Run the second job after the first one finishes
# PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/group_grpo_config.yaml
