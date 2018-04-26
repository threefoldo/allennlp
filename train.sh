#!/bin/bash

rm -rf /tmp/out_$1 && allennlp train training_config/$1.json -s /tmp/out_$1
