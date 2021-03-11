""" Export pair word vocabulary of RELATIVE pretrained embedding. The produced vocabulary file can be downloaded at
https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz
"""
import pickle

from gensim.models import KeyedVectors
from scripts.util import wget

if __name__ == '__main__':
    _url = 'https://drive.google.com/u/0/uc?id=1HVJnTjcaQ3aCLdwTZwiGLpMDyEylx-zS&export=download'
    _path = wget(_url, './cache', gdrive=True)
    model = KeyedVectors.load_word2vec_format(_path, binary=True)
    k = list(map(lambda x: [i.replace('_', ' ') for i in x.split('__')], model.vocab.keys()))
    with open('./relative_vocab.pkl', "wb") as fp:
        pickle.dump(k, fp)
