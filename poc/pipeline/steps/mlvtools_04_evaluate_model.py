#!/usr/bin/env python3
# Generated from ./poc/pipeline/notebooks/04_Evaluate_model.ipynb
from typing import List
import argparse


def mlvtools_04_evaluate_model(model_file: str, data_file: str, result_file: str):
    """
    :param str model_file: Path to model file
    :param str data_file: Path to data files
    :param str result_file: Path to file for storing evaluation metrics
    :dvc-in data_file: ./poc/data/data_train_tokenized.csv 
    :dvc-in model_file: ./poc/data/fasttext_model.bin 
    :dvc-out result_file: ./poc/data/metrics.txt
    """

    # # Evaluate the model
    # Next, we want to evaluate how well the model is doing, on train and test data.

    import pandas as pd
    import numpy as np
    from pyfasttext import FastText
    import json

    df = pd.read_csv(data_file)
    df['data'] = df['data'].apply(lambda s: ' '.join(json.loads(s.replace("'", '"'))))

    model = FastText()
    model.load_model(model_file)

    predicted = pd.DataFrame(model.predict([sentence + '\n' for sentence in df['data']]), columns=['targetnames'])

    accuracy = ((predicted != df[['targetnames']]).sum() / len(df)).iloc[0]

    with open(result_file, 'w') as file_desc:
        file_desc.write(f'accuracy {accuracy}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command for script mlvtools_04_evaluate_model')

    parser.add_argument('--model-file', type=str, required=True, help="Path to model file")

    parser.add_argument('--data-file', type=str, required=True, help="Path to data files")

    parser.add_argument('--result-file', type=str, required=True, help="Path to file for storing evaluation metrics")

    args = parser.parse_args()

    mlvtools_04_evaluate_model(args.model_file, args.data_file, args.result_file)
