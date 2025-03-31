#!/bin/bash

PYTHONPATH=src python3 -m scripts.build_coherent_dataset


PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m scripts.run_train_classifier
