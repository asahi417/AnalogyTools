""" Google analogy test benchmark with word embedding model """
import logging
from random import seed
from gensim.test.utils import datapath
import pandas as pd
from util import get_word_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
seed(1)


def test_analogy(model_type):
    model = get_word_embedding_model(model_type)
    analogy_result = model.evaluate_word_analogies(datapath('questions-words.txt'))
    return {'model_type': model_type, 'accuracy': analogy_result[0]}


if __name__ == '__main__':
    out_fasttext = test_analogy('fasttext')
    out_fasttext_w = test_analogy('fasttext_wiki')
    out_glove = test_analogy('glove')
    out_w2v = test_analogy('w2v')
    full_result = [out_fasttext, out_fasttext_w, out_glove, out_w2v]
    # word embeddings

    out = pd.DataFrame(full_result)
    logging.info('finish evaluation:\n{}'.format(out))
    out.to_csv('./results/google_word_analogy.csv')

