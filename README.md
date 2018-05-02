## Setting up a virtual environment

1.  install pipenv

    ```
    pip install allennlp
    ```

2.  install required packages
    ```
    pipenv install
    ```

That's it! You're now ready to build and train AllenNLP models.


## Train words classifier

1. create a config file for the training
   ```
   cp training_config/ner0416_stage0.json training_config/classifier.json
   ```

2. change 'train_data_path' and 'validation_data_path' in the config

3. train the classifier and save the file 'model.tar.gz' in /tmp/out_classifier
   ```
   ./train.sh classifier
   ```

4. run the web service on port 9000
   ```
   ./serve_classifier.sh model.tar.gz
   ```

## Train lstm_crf model

1. create a config file
   ```
   cp training_config/ner_lstm_crf.json training_config/train.json
   ```

2. change data_path

3. train the model and save the file 'model.tar.gz' in /tmp/out_train
   ```
   ./train.sh train
   ```

4. run the web service on port 9000
   ```
   ./serve_ner.sh model.tar.gz
   ```
