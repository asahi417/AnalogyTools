""" Solve multi choice analogy task by word embedding model """
import os
import logging
import json
from random import randint, seed
from itertools import combinations

import pandas as pd
import numpy as np
from util import wget, get_word_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
seed(1)
# Analogy data
analogy_data = ['sat', 'u2', 'u4', 'google', 'bats']
cache_dir = './cache'
root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0'


def get_dataset_raw(data_name: str):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0'
    assert data_name in analogy_data, 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        wget('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/pmi/pmi.1.csv'.format(cache_dir, data_name), 'r') as f:
        tmp = [_i.split('\t') for _i in f.read().split('\n')]
        d = {'__'.join(i[:2]): float(i[2]) for i in tmp if len(i) == 3 and i[2] not in ['x', 'None']}
    return val_set, test_set, d


def embedding(term, model):
    try:
        return model[term]
    except Exception:
        return None


def cos_similarity(a_, b_):
    if a_ is None or b_ is None:
        return -100
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -100
    return inner / (norm_b * norm_a)


def get_prediction_we(stem, choice, embedding_model, add_feature_set='concat'):

    def diff(vec_a, vec_b):
        if vec_a is None or vec_b is None:
            return None
        if 'concat' in add_feature_set:
            feature = [vec_a, vec_b]
        else:
            feature = []
        if 'diff' in add_feature_set:
            feature.append(vec_a - vec_b)
        if 'dot' in add_feature_set:
            feature.append(vec_a * vec_b)
        assert len(feature)
        return np.concatenate(feature)

    stem_e = diff(embedding(stem[0], embedding_model), embedding(stem[1], embedding_model))
    if stem_e is None:
        return None
    choice_e = [diff(embedding(a, embedding_model), embedding(b, embedding_model)) for a, b in choice]
    score = [cos_similarity(e, stem_e) for e in choice_e]
    _pred = score.index(max(score))
    if score[_pred] == -100:
        return None
    return _pred


def get_prediction_re(stem, choice, embedding_model, lower_case: bool = True):
    if lower_case:
        stem = '__'.join(stem).lower().replace(' ', '_')
        choice = ['__'.join(c).lower().replace(' ', '_') for c in choice]
    else:
        stem = '__'.join(stem).replace(' ', '_')
        choice = ['__'.join(c).replace(' ', '_') for c in choice]

    e_dict = dict([(_i, embedding(_i, embedding_model)) for _i in choice + [stem]])
    score = [cos_similarity(e_dict[stem], e_dict[c]) for c in choice]
    p = score.index(max(score))
    if score[p] == -100:
        return None
    return p


def test_analogy(model_type, relative: bool = False, add_feature_set='concat'):
    model = get_word_embedding_model(model_type)
    if relative:
        get_prediction = get_prediction_re
    else:
        get_prediction = get_prediction_we
    pattern = list(combinations(['diff', 'concat', 'dot'], 2)) + [('diff', 'concat', 'dot')] + ['diff', 'concat', 'dot']
    # prediction = {}
    results = []
    for p in pattern:
        for i in analogy_data:
            tmp_result = {'data': i, 'model': model_type}
            val, test, pmi = get_dataset_raw(i)
            # prediction[i] = {}
            for prefix, data in zip(['test', 'valid'], [test, val]):
                _pred = [get_prediction(o['stem'], o['choice'], model, add_feature_set) for o in data]
                # prediction[i][prefix] = _pred
                tmp_result['oov_{}'.format(prefix)] = len([p for p in _pred if p is None])
                # random prediction when OOV occurs
                _pred = [p if p is not None else randint(0, len(data[n]['choice']) - 1)
                         for n, p in enumerate(_pred)]
                accuracy = sum([o['answer'] == _pred[n] for n, o in enumerate(data)]) / len(_pred)
                tmp_result['accuracy_{}'.format(prefix)] = accuracy
            tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                      tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
            tmp_result['feature'] = p
            results.append(tmp_result)

    return results


if __name__ == '__main__':
    full_result = []
    full_result += test_analogy('glove')
    full_result += test_analogy('w2v')
    full_result += test_analogy('fasttext')
    full_result += test_analogy('fasttext_wiki')
    out = pd.DataFrame(full_result)
    out = out.sort_values(by=['data', 'model'])
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('./analogy_test_mc.results.csv')

