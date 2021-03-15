""" Solve analogy task by word embedding model """
import os
import logging
import json
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
PATH_FASTTEXT_EMBEDDING = './cache/fasttext_diff_vectors.bin'
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


def get_prediction(stem, choice, embedding_model):
    stem = '__'.join(stem).lower().replace(' ', '_')
    choice = ['__'.join(c).lower().replace(' ', '_') for c in choice]
    e_dict = dict([(_i, embedding(_i, embedding_model)) for _i in choice + [stem]])
    score = [cos_similarity(e_dict[stem], e_dict[c]) for c in choice]
    pred = score.index(max(score))
    if score[pred] == -100:
        return None
    return pred


def test_analogy(is_relative, reference_prediction=None):
    if is_relative:
        model_path = PATH_RELATIVE_EMBEDDING
        model_name = 'relative_init'
    else:
        model_path = PATH_FASTTEXT_EMBEDDING
        model_name = 'fasttext_diff'
    word_embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    prediction = {}
    results = []
    for i in DATA:
        tmp_result = {'model': model_name, 'data': i}
        val, test = get_dataset_raw(i)
        prediction[i] = {}
        for prefix, data in zip(['test', 'valid'], [test, val]):
            pred = [get_prediction(o['stem'], o['choice'], word_embedding_model) for o in data]
            prediction[i][prefix] = pred
            tmp_result['oov_{}'.format(prefix)] = len([p for p in pred if p is None])
            pred = [p if p is not None else reference_prediction[i][prefix][n] for n, p in enumerate(pred)]
            accuracy = sum([o['answer'] == pred[n] for n, o in enumerate(data)]) / len(pred)
            tmp_result['accuracy_{}'.format(prefix)] = accuracy
        tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                  tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
        results.append(tmp_result)
    return results, prediction


if __name__ == '__main__':
    import pandas as pd
    # if relative dose not have the pair in its vocabulary, we use diff fasttext's prediction as it doesn't have
    # OOV as its nature.

    results_fasttext, p_fasttext = test_analogy(False)
    with open('./fasttext_prediction.json', 'w') as f_write:
        json.dump(p_fasttext, f_write)
    results_relative, _ = test_analogy(True, p_fasttext)
    out = pd.DataFrame(results_fasttext + results_relative)
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('./result.csv')

