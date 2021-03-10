import os
import os
import pickle
import struct
import logging
import requests
import tarfile
import zipfile

from gensim.models import KeyedVectors


def wget(url, cache_dir):
    filename = os.path.basename(url)
    dst = '{}/{}'.format(cache_dir, filename)
    if os.path.exists(dst):
        logging.debug('found file at `{}`, skip `wget`'.format(dst))
    else:
        with open(dst, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    return dst


def save_pickle(_list, export_dir, chunk_size: int):
    os.makedirs(export_dir, exist_ok=True)
    for s_n, n in enumerate(range(0, len(_list), chunk_size)):
        _list_sub = _list[n:min(n + chunk_size, len(_list))]



def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

# def open_compressed_file(url, cache_dir):
#
#     path = wget(url, cache_dir)
#     if path.endswith('.tar') or path.endswith('.tar.gz') or path.endswith('.tgz'):
#         tar = tarfile.open(path)
#         tar.extractall(cache_dir)
#         tar.close()
#     elif path.endswith('.zip'):
#         with zipfile.ZipFile(path, 'r') as zip_ref:
#             zip_ref.extractall(cache_dir)

_url = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative-init_wikipedia_en_300d.bin'
_path = './cache'
os.makedirs(_path)
_path = wget(_url, _path)
model = KeyedVectors.load_word2vec_format(_path, binary=True)
k = list(map(lambda x: [i.replace('_', ' ') for i in x.split('__')], model.vocab.keys()))

with open('./relative_vocab.pkl', "wb") as fp:
    pickle.dump(k, fp)
