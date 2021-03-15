""" Solve analogy task by word embedding model """
import os
import logging
import json

from gensim.models import fasttext
from gensim.models import KeyedVectors
from util import open_compressed_file

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Relative embedding
URL_RELATIVE_EMBEDDING = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz'
PATH_RELATIVE_EMBEDDING = './cache/relative_init_vectors.bin'
if not os.path.exists(PATH_RELATIVE_EMBEDDING):
    logging.info('downloading relative model')
    open_compressed_file(url=URL_RELATIVE_EMBEDDING, cache_dir='./cache')

# Relative embedding
URL_FASTTEXT_EMBEDDING = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz'
PATH_FASTTEXT_EMBEDDING = './cache/relative_init_vectors.bin'
if not os.path.exists(PATH_FASTTEXT_EMBEDDING):
    logging.info('downloading diff_fasttext model')
    open_compressed_file(url=URL_FASTTEXT_EMBEDDING, cache_dir='./cache')


# Analogy data
DATA = ['sat', 'u2', 'u4', 'google', 'bats']


def get_dataset_raw(data_name: str):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = './cache'
    root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0'
    assert data_name in DATA, 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        open_compressed_file('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
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


def get_prediction(stem, choice, embedding_model, relative: bool = False):
    if relative:
        # relative vector: relative vector can handle lowercase only
        stem = '__'.join(stem).lower()
        choice = ['__'.join(c).lower() for c in choice]
        e_dict = dict([(_i, embedding(_i, embedding_model)) for _i in choice + [stem]])
    else:
        # diff vector
        def diff(x, y):
            if x is None or y is None:
                return None
            return x - y

        e_dict = {
            '__'.join(stem): diff(embedding(stem[0], embedding_model), embedding(stem[1], embedding_model))
        }
        for h, t in choice:
            e_dict['__'.join([h, t])] = diff(embedding(h, embedding_model), embedding(t, embedding_model))
        stem = '__'.join(stem)
        choice = ['__'.join(c) for c in choice]

    score = [cos_similarity(e_dict[stem], e_dict[c]) for c in choice]
    pred = score.index(max(score))
    if score[pred] == -100:
        return None
    return pred


def test_analogy(is_relative, reference_prediction=None):
    if is_relative:
        word_embedding_model = KeyedVectors.load_word2vec_format(PATH_RELATIVE_EMBEDDING, binary=True)
        model_name = 'relative'
    else:
        word_embedding_model = fasttext.load_facebook_model(PATH_WORD_EMBEDDING)
        model_name = 'fasttext'

    prediction = {}
    results = []
    for i in DATA:
        tmp_result = {'model': model_name, 'data': i}
        val, test = get_dataset_raw(i)
        prediction[i] = {}
        for prefix, data in zip(['test', 'valid'], [test, val]):
            pred = [get_prediction(o['stem'], o['choice'], word_embedding_model, relative=is_relative) for o in data]
            prediction[i][prefix] = pred
            tmp_result['oov_{}'.format(prefix)] = len([p for p in pred if p is None])
            pred = [p if p is not None else reference_prediction[i][prefix][n] for n, p in enumerate(pred)]
            accuracy = sum([o['answer'] == pred[n] for n, o in enumerate(data)]) / len(pred)
            tmp_result['accuracy_{}'.format(prefix)] = accuracy
        results.append(tmp_result)
    return results, prediction


if __name__ == '__main__':
    import pandas as pd
    results_fasttext, p_fasttext = test_analogy(False)
    results_relative, _ = test_analogy(True, p_fasttext)
    out = pd.DataFrame(results_fasttext + results_relative)
    print(out)

    with open('./result.jsonl', 'w') as f:
        for line in results_fasttext + results_relative:
            f.write(json.dumps(line) + '\n')
    logging.info('finish evaluation: result was exported to {}'.format('./result.jsonl'))