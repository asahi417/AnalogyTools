""" Solve analogy task by word embedding model """
import os
import logging
import json
from itertools import chain
from random import randint

import pandas as pd
from gensim.models import fasttext
from util import open_compressed_file

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Anchor word embedding model
URL_WORD_EMBEDDING = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
PATH_WORD_EMBEDDING = './cache/crawl-300d-2M-subword.bin'
if not os.path.exists(PATH_WORD_EMBEDDING):
    logging.info('downloading fasttext model')
    open_compressed_file(url=URL_WORD_EMBEDDING, cache_dir='./cache')

# Relative embedding
URL_RELATIVE_EMBEDDING = ''
PATH_RELATIVE_EMBEDDING = './cache/relative_init_vectors.txt'
if not os.path.exists(PATH_WORD_EMBEDDING):
    logging.info('downloading relative model')
    open_compressed_file(url=URL_RELATIVE_EMBEDDING, cache_dir='./cache')


def get_analogy_data():
    """ Get analogy data """
    data_url = dict(
        sat='https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/sat.zip',
        u2='https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u2.zip',
        u4='https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u4.zip',
        google='https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/google.zip',
        bats='https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/bats.zip')

    def extract_pairs(_json_file):
        _json = json.loads(_json_file)
        return [[a.lower() for a in _json['stem']]] + [[a.lower(), b.lower()] for a, b in _json['choice']]

    for name, url in data.items():
        open_compressed_file(url=url, cache_dir='./cache')
        with open('./cache/{}/valid.jsonl'.format(name), 'r') as f_:
            pairs = list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
        with open('./cache/{}/test.jsonl'.format(name), 'r') as f_:
            pairs += list(chain(*[extract_pairs(i) for i in f_.read().split('\n') if len(i) > 0]))
    return pairs


def get_dataset_raw(data_name: str, cache_dir: str = default_cache_dir_analogy):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats'], 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        util.open_compressed_file('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val, test



def embedding(term, model):
    try:
        return model[term]
    except Exception:
        return None


def cos_similarity(a_, b_):
    if a_ is None or b_ is None:
        return DUMMY
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    if norm_a == 0 or norm_b == 0:
        return DUMMY
    return inner / (norm_b * norm_a)


def get_embedding(word_list, model):
    embeddings = [(_i, embedding(_i, model)) for _i in word_list]
    embeddings = list(filter(lambda x: x[1] is not None, embeddings))
    return dict(embeddings)


def get_prediction(stem, choice, embedding_dict):
    def diff(x, y):
        if x is None or y is None:
            return None
        return x - y

    stem_e = diff(embedding(stem[0], embedding_dict), embedding(stem[1], embedding_dict))
    if stem_e is None:
        return None
    choice_e = [diff(embedding(a, embedding_dict), embedding(b, embedding_dict)) for a, b in choice]
    score = [cos_similarity(e, stem_e) for e in choice_e]
    pred = score.index(max(score))
    if score[pred] == DUMMY:
        return None
    return pred


if __name__ == '__main__':
    for prefix in ['test', 'valid']:
        line_oov = []
        line_accuracy = []
        for i in DATA:
            val, test = get_dataset_raw(i)
            data = test if prefix == 'test' else val
            oov = {'data': i}
            all_accuracy = {'data': i}
            answer = {n: o['answer'] for n, o in enumerate(data)}
            random_prediction = {n: randint(0, len(o['choice']) - 1) for n, o in enumerate(data)}
            all_accuracy['random'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)

            vocab = list(set(list(chain(*[list(chain(*[o['stem']] + o['choice'])) for o in data]))))

            dict_ = get_embedding(vocab, model_ft)
            ft_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(data)}

            all_accuracy['fasttext'] = sum([answer[n] == ft_prediction[n] for n in range(len(answer))]) / len(answer)
            line_accuracy.append(all_accuracy)

        pd.DataFrame(line_accuracy).to_csv(
            'experiments_results/summary/experiment.word_embedding.{}.csv'.format(prefix))
        pd.DataFrame(line_oov).to_csv(
            'experiments_results/summary/experiment.word_embedding.{}.oov.csv'.format(prefix))