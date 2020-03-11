#!/usr/bin/env python3
# Generated from ./poc/pipeline/notebooks/03_Classify_text.ipynb
from typing import List
import argparse


def mlvtools_03_classify_text(input_csv_file: str, out_model_path: str, learning_rate: float, epochs: int):
    """
    :param str input_csv_file: Path to input file
    :param str out_model_path: Path to model files
    :param float learning_rate: Learning rate
    :param int epochs: Number of epochs
    
    :dvc-in input_csv_file: ./poc/data/data_train_tokenized.csv
    :dvc-out out_model_path: ./poc/data/fasttext_model.bin
    :dvc-out: ./poc/data/fasttext_model.vec
    :dvc-extra: --learning-rate 0.7 --epochs 20
    """

    # # Classify text
    # We are going to train a classifier on the tokenized text input, using the [FastText libary](https://fasttext.cc/).
    #
    # In addition to the input data file, we give to the command a few hyperparameter values, and we store the binary file representing the learned model as output.
    #
    # We only learn for a few epochs, to see how the versioning tools work.

    import pandas as pd
    import numpy as np
    from collections import Counter
    from pyfasttext import FastText
    import tempfile
    import os

    df = pd.read_csv(input_csv_file)

    import json
    df['data'] = df['data'].apply(lambda s: json.loads(s.replace("'", '"')))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, 'unigrams')
        with open(tmp_path, 'w') as f:
            for text, _, lab in df.itertuples(index=False, name=None):
                f.write('__label__{} {}\n'.format(lab, ' '.join(text)))

        model = FastText()
        # Fastext automatically add .bin at the end of the output model file name
        out_model_path = out_model_path.replace('.bin', '')
        model.supervised(input=tmp_path, output=out_model_path, epoch=epochs, lr=learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command for script mlvtools_03_classify_text')

    parser.add_argument('--input-csv-file', type=str, required=True, help="Path to input file")

    parser.add_argument('--out-model-path', type=str, required=True, help="Path to model files")

    parser.add_argument('--learning-rate', type=float, required=True, help="Learning rate")

    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")

    args = parser.parse_args()

    mlvtools_03_classify_text(args.input_csv_file, args.out_model_path, args.learning_rate, args.epochs)
