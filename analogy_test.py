""" Solve multi choice analogy task by word embedding model """
import os
import logging
import json
from itertools import combinations

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


def get_prediction_we(stem, choice, embedding_model, add_feature_set='concat', relative_model=None):

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

    if relative_model is not None:
        stem_e_r = embedding('__'.join(stem).lower().replace(' ', '_'), relative_model)
        if stem_e_r is not None:
            stem_e = np.concatenate([stem_e,  stem_e_r])
            choice_e_r = [embedding('__'.join(c).lower().replace(' ', '_'), relative_model) for c in choice]
            choice_e = [np.concatenate([a, b]) if a is not None and b is not None else None
                        for a, b in zip(choice_e, choice_e_r)]

    score = [cos_similarity(e, stem_e) for e in choice_e]
    _pred = score.index(max(score))
    if score[_pred] == -100:
        return None
    return _pred


def test_analogy(model_type, add_relative: bool = False):
    model = get_word_embedding_model(model_type)
    model_re = None
    if add_relative:
        model_re = get_word_embedding_model('relative_init.{}'.format(model_type))
    pattern = list(combinations(['diff', 'concat', 'dot'], 2)) + [('diff', 'concat', 'dot')] + ['diff', 'concat', 'dot']
    results = []

    for _pattern in pattern:
        for i, (val, test) in full_data.items():
            tmp_result = {'data': i, 'model': model_type, 'add_relative': add_relative}
            for prefix, data in zip(['test', 'valid'], [test, val]):
                _pred = [get_prediction_we(o['stem'], o['choice'], model, _pattern, relative_model=model_re)
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
        results.append(tmp_result)
    return results


if __name__ == '__main__':
    full_result = pmi_baseline()
    full_result += test_analogy('glove')
    full_result += test_analogy('w2v')
    full_result += test_analogy('fasttext')
    full_result += test_analogy('glove', add_relative=True)
    full_result += test_analogy('w2v', add_relative=True)
    full_result += test_analogy('fasttext', add_relative=True)
    out = pd.DataFrame(full_result)
    out = out.sort_values(by=['data', 'model'])
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('results/analogy_test.csv')

