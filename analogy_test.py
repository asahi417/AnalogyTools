""" Solve analogy task by word embedding model """
import os
import logging
import json
from random import randint, seed

import pandas as pd
from util import wget, get_word_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
seed(1)
# Analogy data
DATA = ['sat', 'u2', 'u4', 'google', 'bats']


def get_dataset_raw(data_name: str):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = './cache'
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0'
    assert data_name in DATA, 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        wget('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val_set, test_set


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


def get_prediction_we(stem, choice, embedding_model):

    def diff(x, y):
        if x is None or y is None:
            return None
        return x - y

    stem_e = diff(embedding(stem[0], embedding_model), embedding(stem[1], embedding_model))
    if stem_e is None:
        return None
    choice_e = [diff(embedding(a, embedding_model), embedding(b, embedding_model)) for a, b in choice]
    score = [cos_similarity(e, stem_e) for e in choice_e]
    pred = score.index(max(score))
    if score[pred] == -100:
        return None
    return pred


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


def test_analogy(model_type, relative: bool = False):
    model = get_word_embedding_model(model_type)
    if relative:
        get_prediction = get_prediction_re
    else:
        get_prediction = get_prediction_we

    prediction = {}
    results = []
    for i in DATA:
        tmp_result = {'data': i, 'model': model_type}
        val, test = get_dataset_raw(i)
        prediction[i] = {}
        for prefix, data in zip(['test', 'valid'], [test, val]):
            pred = [get_prediction(o['stem'], o['choice'], model) for o in data]
            prediction[i][prefix] = pred
            tmp_result['oov_{}'.format(prefix)] = len([p for p in pred if p is None])
            # random prediction when OOV occurs
            pred = [p if p is not None else randint(0, len(data[n]['choice']) - 1)
                    for n, p in enumerate(pred)]
            accuracy = sum([o['answer'] == pred[n] for n, o in enumerate(data)]) / len(pred)
            tmp_result['accuracy_{}'.format(prefix)] = accuracy
        tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                  tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
        results.append(tmp_result)
    return results, prediction


if __name__ == '__main__':

    full_result = []
    os.makedirs('./predictions', exist_ok=True)

    # relative embeddings (concat)
    tmp_results, pred = test_analogy('relative_init.fasttext.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.fasttext.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('relative_init.glove.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.glove.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('relative_init.w2v.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.w2v.json', 'w') as f_write:
        json.dump(pred, f_write)

    # relative embeddings (concat & truecase)
    tmp_results, pred = test_analogy('relative_init.fasttext.truecase.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.fasttext.truecase.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('relative_init.glove.truecase.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.glove.truecase.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('relative_init.w2v.truecase.concat', relative=True)
    full_result += tmp_results
    with open('./predictions/relative_init.w2v.truecase.json', 'w') as f_write:
        json.dump(pred, f_write)

    # word embeddings
    tmp_results, pred = test_analogy('glove')
    full_result += tmp_results
    with open('./predictions/glove.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('w2v')
    full_result += tmp_results
    with open('./predictions/w2v.json', 'w') as f_write:
        json.dump(pred, f_write)

    tmp_results, pred = test_analogy('fasttext')
    full_result += tmp_results
    with open('./predictions/fasttext.json', 'w') as f_write:
        json.dump(pred, f_write)

    out = pd.DataFrame(full_result)
    out = out.sort_values(by=['data', 'model'])
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('./analogy_test_baseline.csv')

