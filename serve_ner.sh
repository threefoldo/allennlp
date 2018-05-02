#!/bin/bash

# python -m allennlp.service.server_simple --archive-path /tmp/ner0419_stage0/model.tar.gz --predictor article_predictor --title 'NER classifier' --field-name word
python -m allennlp.service.server_simple --archive-path trained/out_ner_lstm_crf/model.tar.gz --predictor sentence-tagger --port 9000 --title 'NER' --field-name sentence
