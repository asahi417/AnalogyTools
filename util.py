""" Download files from web. If Gdrive downloading stacked, try to remove the gdown cache `rm -rf ~/.cache/gdown` """
import tarfile
import zipfile
import requests
import os
import json
import pickle
from itertools import chain

import gdown
from gensim.models import KeyedVectors


def open_compressed_file(url, cache_dir, filename: str = None, gdrive: bool = False):
    path = wget(url, cache_dir, gdrive=gdrive, filename=filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def wget(url, cache_dir, gdrive: bool = False, filename: str = None):
    os.makedirs(cache_dir, exist_ok=True)
    if gdrive:
        if filename:
            return gdown.download(url, '{}/{}'.format(cache_dir, filename), quiet=False)
        else:
            return gdown.download(url, cache_dir, quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)


def get_embedding_model(model_type: str = 'relative_init', cache_dir: str = './cache'):
    root_url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/'
    urls = {
        'relative_init': root_url + 'relative_init_vectors.bin.tar.gz',
        'fasttext_diff': root_url + 'fasttext_diff_vectors.bin.tar.gz',
        'concat_relative_fasttext': 'https://drive.google.com/u/0/uc?id=1CkdsxEl21TUiBmLS6uq55tH6SiHvWGDn&export=download'
    }
    assert model_type in urls, '{} not in {}'.format(model_type, urls.keys())
    model_name = model_type + '_vectors.bin'

    model_path = '{}/{}'.format(cache_dir, model_name)
    if not os.path.exists(model_path):
        open_compressed_file(url=urls[model_type], cache_dir=cache_dir, gdrive=model_type == 'concat_relative_fasttext',
                             filename=model_name+'.tar.gz')
    word_embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_embedding_model


def get_pair_relative(cache_dir: str = './cache'):
    """ Get the list of word pairs in RELATIVE pretrained model """
    path = '{}/relative-init_wikipedia_en_300d.bin'.format(cache_dir)
    if not os.path.exists(path):
        url = 'https://drive.google.com/u/0/uc?id=1HVJnTjcaQ3aCLdwTZwiGLpMDyEylx-zS&export=download'
        wget(url, cache_dir, gdrive=True, filename='relative-init_wikipedia_en_300d.bin')
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    return list(map(lambda x: [i.replace('_', ' ') for i in x.split('__')], model.vocab.keys()))


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
        open_compressed_file(url=url, cache_dir=cache_dir)
        with open('./cache/{}/valid.jsonl'.format(name), 'r') as f_:
            pairs += list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
        with open('./cache/{}/test.jsonl'.format(name), 'r') as f_:
            pairs += list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
    return pairs


def get_common_word_pair(cache_dir: str = './cache'):
    url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl.tar.gz'
    open_compressed_file(url=url, cache_dir=cache_dir)
    with open('{}/common_word_pairs.pkl'.format(cache_dir), "rb") as fp:
        return pickle.load(fp)
