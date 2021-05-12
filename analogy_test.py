""" Solve multi choice analogy task by word embedding model """
import os
import logging
import json

import pandas as pd
import numpy as np
from util import wget, get_word_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_analogy_data():
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.tar.gz'
    if not os.path.exists('{}/analogy_test_dataset'.format(cache_dir)):
        wget(root_url_analogy, cache_dir)
    data = {}
    for d in ['bats', 'sat', 'u2', 'u4', 'google']:
        with open('{}/analogy_test_dataset/{}/test.jsonl'.format(cache_dir, d), 'r') as f:
            test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
        with open('{}/analogy_test_dataset/{}/valid.jsonl'.format(cache_dir, d), 'r') as f:
            val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
        data[d] = (val_set, test_set)
    return data


full_data = get_analogy_data()


def embedding(term, model):
    if model is None:
        return np.zeros(3)
    try:
        return model[term]
    except KeyError:
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


def get_prediction_we(stem, choice, embedding_model, add_feature_set='concat',
                      relative_model=None, pair2vec_model=None, bi_direction: bool = False):

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

    def get_pair_embedding(_stem_e, _choice_e, pair_model, if_reverse: bool = False):
        if if_reverse:
            _stem = [stem[1], stem[0]]
            _choice = [[_i[1], _i[0]] for _i in choice]
        else:
            _stem = stem
            _choice = choice

        stem_e_r = embedding('__'.join(_stem).lower().replace(' ', '_'), pair_model)
        if stem_e_r is not None:
            _stem_e = np.concatenate([_stem_e, stem_e_r])
            choice_e_r = [embedding('__'.join(c).lower().replace(' ', '_'), pair_model) for c in _choice]
            _choice_e = [np.concatenate([a, b]) if a is not None and b is not None else None
                         for a, b in zip(_choice_e, choice_e_r)]
        return _stem_e, _choice_e

    for _pair_model in [pair2vec_model, relative_model]:
        if _pair_model is None:
            continue
        stem_e, choice_e = get_pair_embedding(stem_e, choice_e, _pair_model)
        if bi_direction:
            stem_e, choice_e = get_pair_embedding(stem_e, choice_e, _pair_model, if_reverse=True)

    score = [cos_similarity(e, stem_e) for e in choice_e]
    _pred = score.index(max(score))
    if score[_pred] == -100:
        return None
    return _pred


def test_analogy(model_type, add_relative: bool = False, add_pair2vec: bool = False, bi_direction: bool = False,
                 only_pair_embedding: bool = False):

    model_re = None
    model_p2v = None
    if only_pair_embedding:
        model = None
    else:
        model = get_word_embedding_model(model_type)
    if add_relative:
        model_re = get_word_embedding_model('relative_init.{}'.format(model_type))
    if add_pair2vec:
        model_p2v = get_word_embedding_model('pair2vec')
    if only_pair_embedding:
        assert model_p2v or model_re
    else:
        assert model

    if only_pair_embedding:
        pattern = ['concat']
    else:
        pattern = ['diff', 'concat', ('diff', 'dot'), ('concat', 'dot')]
    results = []

    for _pattern in pattern:
        for i, (val, test) in full_data.items():
            tmp_result = {'data': i, 'model': model_type, 'add_relative': add_relative, 'add_pair2vec': add_pair2vec,
                          'bi_direction': bi_direction, 'only_pair_embedding': only_pair_embedding}
            for prefix, data in zip(['test', 'valid'], [test, val]):
                _pred = [get_prediction_we(o['stem'], o['choice'], model, _pattern, relative_model=model_re,
                                           pair2vec_model=model_p2v, bi_direction=bi_direction)
                         for o in data]
                tmp_result['oov_{}'.format(prefix)] = len([p for p in _pred if p is None])
                # random prediction when OOV occurs
                _pred = [p if p is not None else data[n]['pmi_pred'] for n, p in enumerate(_pred)]
                accuracy = sum([o['answer'] == _pred[n] for n, o in enumerate(data)]) / len(_pred)
                tmp_result['accuracy_{}'.format(prefix)] = accuracy
            tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                      tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
            tmp_result['feature'] = _pattern
            results.append(tmp_result)

    return results


def pmi_baseline():
    results = []
    for i, (val, test) in full_data.items():
        tmp_result = {'data': i, 'model': 'PMI'}
        for prefix, data in zip(['test', 'valid'], [test, val]):
            tmp_result['oov_{}'.format(prefix)] = 0
            accuracy = sum([o['answer'] == o['pmi_pred'] for n, o in enumerate(data)]) / len(data)
            tmp_result['accuracy_{}'.format(prefix)] = accuracy
        tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                  tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
        tmp_result['feature'] = None
        tmp_result['add_relative'] = None
        results.append(tmp_result)
    return results


if __name__ == '__main__':
    full_result = pmi_baseline()

    full_result += test_analogy('fasttext', add_pair2vec=True, bi_direction=True, only_pair_embedding=True)
    full_result += test_analogy('fasttext', add_relative=True, bi_direction=True, only_pair_embedding=True)

    full_result += test_analogy('fasttext', add_pair2vec=True, bi_direction=True)
    full_result += test_analogy('fasttext', add_pair2vec=True)
    full_result += test_analogy('fasttext', add_relative=True, bi_direction=True)
    full_result += test_analogy('fasttext', add_relative=True)
    full_result += test_analogy('fasttext')

    full_result += test_analogy('glove', add_pair2vec=True, bi_direction=True)
    full_result += test_analogy('glove', add_pair2vec=True)
    full_result += test_analogy('glove', add_relative=True, bi_direction=True)
    full_result += test_analogy('glove', add_relative=True)
    full_result += test_analogy('glove')

    full_result += test_analogy('w2v', add_pair2vec=True, bi_direction=True)
    full_result += test_analogy('w2v', add_pair2vec=True)
    full_result += test_analogy('w2v', add_relative=True, bi_direction=True)
    full_result += test_analogy('w2v', add_relative=True)
    full_result += test_analogy('w2v')

    out = pd.DataFrame(full_result)
    out = out.sort_values(by=['data', 'model'])
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('results/analogy_test.csv')

