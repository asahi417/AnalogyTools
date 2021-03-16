""" Download files from web. If Gdrive downloading stacked, try to remove the gdown cache `rm -rf ~/.cache/gdown` """
import tarfile
import zipfile
import requests
import os

import gdown
from gensim.models import KeyedVectors

__all__ = ('open_compressed_file', 'wget', 'get_embedding_model')


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
    root_url = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/'
    urls = {
        'relative_init': root_url + 'relative_init_vectors.bin.tar.gz',
        'fasttext_diff': root_url + 'fasttext_diff_vectors.bin.tar.gz',
        'concat_relative_fasttext': 'https://drive.google.com/u/0/uc?id=1CkdsxEl21TUiBmLS6uq55tH6SiHvWGDn&export=download'
    }
    assert model_type in urls, '{} not in {}'.format(model_type, urls.keys())
    model_name = model_type + '_vectors.bin'

    model_path = '{}/{}'.format(cache_dir, model_name)
    if not os.path.exists(model_path):
        open_compressed_file(url=urls[model_type], cache_dir=cache_dir, gdrive=model_type=='concat_relative_fasttext',
                             filename=model_name+'.tar.gz')
    word_embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_embedding_model

