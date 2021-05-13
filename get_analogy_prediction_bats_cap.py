""" Solve multi choice analogy task by word embedding model """
import logging
import json

from util import get_word_embedding_model
from analogy_test import get_analogy_data, get_prediction_we

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

full_data = get_analogy_data()


def cap(_list):
    return [i.capitalize() for i in _list]


if __name__ == '__main__':
    _, test = full_data['bats']

    for model_type in ['fasttext', 'glove', 'w2v']:
        model = get_word_embedding_model(model_type)

        _pred = [get_prediction_we(cap(o['stem']), cap(o['choice']), model, 'diff') for o in test]
        for d, p in zip(test, _pred):
            d['pred/{}'.format(model_type)] = p
    with open('results/analogy.prediction.bats.cap.json', 'w') as f:
        json.dump(test, f)

