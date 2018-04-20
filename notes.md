

## stage0

### train the model with noisy dictionary

```
rm -rf /tmp/ner0416_stage0 && python -m allennlp.run train training_config/ner0416_stage0.json -s /tmp/ner0416_stage0
```

### predict another dataset


```
python -m allennlp.run predict /tmp/ner0416_stage0/model.tar.gz --predictor article_predictor data/ner0419_train.jsonl > ~/work/ner/data/ner0419_train_pred.txt
```

### align predictions and labels

```
python scripts/parse_predictions.py ~/work/ner/data/ner0419_train_pred.txt  ner0419_stage0_words.jsonl
```

## next

1. compare two models
  trained on ner0419 (acc: 0.87) ==> trained on filtered ner0419 (acc: ?)

2. if filtered model is better
  - use it to predict on ner0416 and ner0420;
  - combine filtered ner0416, ner0419, ner0420;
  - train the final model
  
