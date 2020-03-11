#!/usr/bin/env python3
from os.path import dirname, join

from sklearn.datasets import fetch_20newsgroups

cache_path = join(dirname(__file__), 'poc', 'data')

buffer = fetch_20newsgroups(data_home=cache_path)
