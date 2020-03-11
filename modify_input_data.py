#!/usr/bin/env python3
import codecs
from os.path import dirname, join
import pickle

from sklearn.datasets._twenty_newsgroups import CACHE_NAME  # noqa - this should be exposed
from sklearn.utils import shuffle

cache_path = join(dirname(__file__), 'poc', 'data', CACHE_NAME)


def shuffle_data(subset: str, cache):
    cache[subset].data, cache[subset].target, cache[subset].filenames = shuffle(cache[subset].data,
                                                                                cache[subset].target,
                                                                                cache[subset].filenames)


with open(cache_path, 'rb') as f:
    compressed_content = f.read()
uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
cache = pickle.loads(uncompressed_content)

shuffle_data('train', cache)
shuffle_data('test', cache)

compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
with open(cache_path, 'wb') as f:
    f.write(compressed_content)
