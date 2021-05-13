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
    full_data['bats_cap'] = full_data['bats']

    for i, (val, test) in full_data.items():
        for data in [test, val]:
            for model_type in ['fasttext', 'glove', 'w2v']:
                model = get_word_embedding_model(model_type)
                if i == 'bats_cap':
                    _pred = [get_prediction_we(cap(o['stem']), cap(o['choice']), model, 'diff') for o in test]
                else:
                    _pred = [get_prediction_we(o['stem'], o['choice'], model, 'diff') for o in data]
                for d, p in zip(data, _pred):
                    d['pred/{}'.format(model_type)] = p

    with open('results/analogy.prediction.json', 'w') as f:
        json.dump(full_data, f)


