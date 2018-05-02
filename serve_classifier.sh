#!/bin/bash

model_file=$1

python -m allennlp.service.server_simple --archive-path ${model_file} --predictor article_predictor --port 9000 --title 'words classifier' --field-name word
