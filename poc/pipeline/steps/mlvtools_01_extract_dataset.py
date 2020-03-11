#!/usr/bin/env python3
# Generated from ./poc/pipeline/notebooks/01_Extract_dataset.ipynb
from typing import List
import argparse


def mlvtools_01_extract_dataset(subset: str, data_home: str, output_path: str):
    """
    :param str subset: Subset of data to load
    :param str data_home: Path to parent directory to cache file
    :param str output_path: Path to output file
    :dvc-cmd: dvc run -f $MLV_DVC_META_FILENAME -o ./poc/data/data_train.csv -o ./poc/data/data_test.csv
                    -d ./poc/data/20news-bydate_py3.pkz
           "$MLV_PY_CMD_PATH --subset train
                --data-home ./poc/data --output-path ./poc/data/data_train.csv &&
            $MLV_PY_CMD_PATH --subset test
                --data-home ./poc/data --output-path ./poc/data/data_test.csv"
    """

    # # Extract train and test data set from 20newsgroups

    # This pipeline step is based on the 02_Extract_dataset.ipynb Jupiter Notebook. It loads data downloaded in the previous step and splits it into a train or a test data set.

    # ### Parameters
    #
    # The following cell (the first code cell) contains a Docstring description of parameters of this step.
    #
    # **param** is used to declare parameters for the corresponding Python 3 script and command.
    #
    #
    # | **param**  |  -- | impact  |
    # | :---         |     :---      |          :--- |
    # |python script | the method wrapping the following code accepts parameters | **subset:str, data_home:str, output_path:str**|
    # |python command line  | it will accepts  | **--subset ['train'\|'test'] --data-home [path] --output-path [path]**|
    #
    #
    # **dvc-cmd** is used here because we want to run the python command twice. **dvc-run** allow to describe the whole DVC command when it is not generic.
    # In this case it is possible (and strongly recommended) to use the variable **$MLV_PY_CMD_PATH** to designate the python command line path.
    #
    #
    # To have a better understanding of those parameters, see the MLV-Tools [documentation](https://github.com/peopledoc/ml-versioning-tools) and have a look to the corresponding generated DVC command line.

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset=subset,
                                          data_home=data_home,
                                          download_if_missing=False,
                                          remove=('headers', 'footers', 'quotes'))

    df_train = pd.DataFrame(newsgroups_train.data, columns=['data'])

    df_train['target'] = newsgroups_train.target

    df_train['targetnames'] = df_train['target'].apply(lambda n: newsgroups_train.target_names[n])

    df_train.to_csv(output_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command for script mlvtools_01_extract_dataset')

    parser.add_argument('--subset', type=str, required=True, help="Subset of data to load")

    parser.add_argument('--data-home', type=str, required=True, help="Path to parent directory to cache file")

    parser.add_argument('--output-path', type=str, required=True, help="Path to output file")

    args = parser.parse_args()

    mlvtools_01_extract_dataset(args.subset, args.data_home, args.output_path)
