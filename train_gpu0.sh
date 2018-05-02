#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
rm -rf /tmp/out_$1 && allennlp train training_config/$1.json -s /tmp/out_$1
