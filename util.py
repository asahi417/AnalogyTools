""" Download files from web. If Gdrive downloading stacked, try to remove the gdown cache `rm -rf ~/.cache/gdown` """
import tarfile
import zipfile
import gzip
import requests
import os
import json
import pickle
from itertools import chain

import gdown
from gensim.models import KeyedVectors
from gensim.models import fasttext


def get_word_embedding_model(model_name: str = 'fasttext'):
    """ get word embedding model """
    os.makedirs('./cache', exist_ok=True)
    if model_name == 'w2v':
        path = './cache/GoogleNews-vectors-negative300.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url="https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download",
                cache_dir='./cache',
                gdrive_filename='GoogleNews-vectors-negative300.bin.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'fasttext':
        path = './cache/crawl-300d-2M-subword.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip',
                cache_dir='./cache')
        model = fasttext.load_facebook_model(path)
    elif model_name == 'glove':
        path = './cache/glove.840B.300d.gensim.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://drive.google.com/u/0/uc?id=1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN&export=download',
                cache_dir='./cache',
                gdrive_filename='glove.840B.300d.gensim.bin.tar.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.concat':
        path = './cache/relative_init.fasttext.concat.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url='https://drive.google.com/u/0/uc?id=1EH0oywBo8OaNExyc5XTGIFhLvf8mZiBz&export=download',
                cache_dir='./cache',
                gdrive_filename='relative_init.fasttext.concat.bin.tar.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.truecase.concat':
        path = './cache/relative_init.fasttext.truecase.concat.bin'
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(
                url="https://drive.google.com/u/0/uc?id=1iUuCYM_UJ6FHI5yxg5UIGkXN4qqU5S3G&export=download",
                cache_dir='./cache',
                gdrive_filename='relative_init.fasttext.truecase.concat.bin.tar.gz'
            )
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        path = './cache/{}.bin'.format(model_name)
        if not os.path.exists(path):
            print('downloading {}'.format(model_name))
            wget(url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/{}.bin.tar.gz'.format(model_name),
                 cache_dir='./cache')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    path = _wget(url, cache_dir, gdrive_filename=gdrive_filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())

    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    # return path


def _wget(url: str, cache_dir, gdrive_filename: str = None):
    """ get data from web """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)


def get_pair_analogy(cache_dir: str = './cache'):
    """ Get the list of word pairs in analogy dataset """
    data = dict(
        sat='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/sat.zip',
        u2='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/u2.zip',
        u4='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/u4.zip',
        google='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/google.zip',
        bats='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/bats.zip'
    )

    def extract_pairs(_json_file):
        _json = json.loads(_json_file)
        return [_json['stem']] + _json['choice']

    pairs = []
    for name, url in data.items():
        wget(url=url, cache_dir=cache_dir)
        with open('./cache/{}/valid.jsonl'.format(name), 'r') as f_:
            pairs += list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
        with open('./cache/{}/test.jsonl'.format(name), 'r') as f_:
            pairs += list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
    return pairs


def get_common_word_pair(cache_dir: str = './cache', if_truecase: bool = False):
    if if_truecase:
        url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs_truecase.pkl.tar.gz'
        path = '{}/common_word_pairs_truecase.pkl'.format(cache_dir)
    else:
        url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl.tar.gz'
        path = '{}/common_word_pairs.pkl'.format(cache_dir)
    wget(url=url, cache_dir=cache_dir)
    with open(path, "rb") as fp:
        return pickle.load(fp)
