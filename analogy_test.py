""" Solve multi choice analogy task by word embedding model """
import os
import logging
import json
from random import randint, seed
from typing import Dict, List, Any, Union

import pandas as pd
from util import wget, get_word_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
seed(1)


def get_google_analogy_test():
    """ get google analogy test """
    cache_dir = './cache'
    url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/google_analogy_test.json'
    data_name = 'google_analogy_test.json'
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        wget(url, cache_dir)
    with open('{}/{}'.format(cache_dir, data_name), 'r') as f:
        data = json.load(f)
    return data


def test_analogy(model_type):
    model = get_word_embedding_model(model_type)
    data = get_google_analogy_test()
    accuracy = {'model_type': model_type}
    out_all = []
    for k, parent_values in data.items():
        logging.info('\t * data: {}'.format(k))
        _out = []
        for v in parent_values:
            head_stem, tail_stem, head, tail = v
            v = model[head] + model[tail_stem] - model[head_stem]
            pred, score = model.similar_by_vector(v)[0]
            _out.append(pred == tail)
        accuracy[k] = sum(_out)/len(_out) * 100
        print('\t * {}: {}'.format(k, accuracy[k]))
        out_all += _out
    accuracy['all'] = sum(out_all)/len(out_all) * 100
    print('\t * all: {}'.format(accuracy['all']))
    return accuracy


if __name__ == '__main__':
    full_result = [
        test_analogy('glove'),
        test_analogy('w2v'),
        test_analogy('fasttext')
    ]
    # word embeddings

    out = pd.DataFrame(full_result)
    # out = out.sort_values(by=['data', 'model'])
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('./analogy_test.result.csv')

