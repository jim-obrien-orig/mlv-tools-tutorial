#!/usr/bin/env python3
# Generated from ./poc/pipeline/notebooks/02_Tokenize_text.ipynb
from typing import List
import argparse


def mlvtools_02_tokenize_text(input_csv_file: str, output_csv_file: str):
    """
    :param str input_csv_file: Path to input file
    :param str output_csv_file: Path to output file
    :dvc-in input_csv_file: ./poc/data/data_train.csv
    :dvc-out output_csv_file: ./poc/data/data_train_tokenized.csv
    """

    # # Tokenize text
    # The next step in the pipeline is to tokenize the text input, as is usual in Natural Language Processing. In order to do that, we use the word punkt tokenizer provided by NLTK.
    #
    # We also remove english stopwords (frequent words who add no semantic meaning, such as "and", "is", "the"...).
    #
    # Each token is also converted to lower-case and non-alphabetic tokens are removed.
    #
    # In this very simple tutorial example, we do not apply any lemmatization technique.

    import pandas as pd
    import numpy as np
    from nltk.tokenize import wordpunct_tokenize
    from nltk.corpus import stopwords

    df = pd.read_csv(input_csv_file)
    df.head()

    stopswords_english = set(stopwords.words('english'))

    def tokenize_and_clean_text(s):
        return [
            token.lower() for token in wordpunct_tokenize(s)
            if token.isalpha() and token.lower() not in stopswords_english
        ]

    df = df.dropna()

    df['data'] = df['data'].apply(tokenize_and_clean_text)

    df.to_csv(output_csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command for script mlvtools_02_tokenize_text')

    parser.add_argument('--input-csv-file', type=str, required=True, help="Path to input file")

    parser.add_argument('--output-csv-file', type=str, required=True, help="Path to output file")

    args = parser.parse_args()

    mlvtools_02_tokenize_text(args.input_csv_file, args.output_csv_file)
