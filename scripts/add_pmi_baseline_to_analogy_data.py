""" Add statistical baseline to analogy test dataset. """
import logging
import json
from random import randint, seed
from analogy_test import get_dataset_raw

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
seed(1)


def add_pmi_baseline(_data, pmi_dict):
    for _d in _data:
        tmp = ['__'.join(k) for k in _d['choice']]
        pmi_score = [pmi_dict[k] if k in pmi_dict else -100 for k in tmp]
        index = [n for n, k in enumerate(pmi_score) if k == max(pmi_score)]
        _d['pmi_pred'] = index[randint(0, len(index) - 1)]
    return sum([_d['pmi_pred'] == _d['answer'] for _d in _data])/len(_data)


if __name__ == '__main__':
    cache_dir = '../cache'
    for da in ['sat', 'u2', 'u4', 'google', 'bats']:
        v, t, pmi = get_dataset_raw(da)
        a_v = add_pmi_baseline(v, pmi)
        a_t = add_pmi_baseline(t, pmi)
        print('- data {}'.format(da))
        print('\t - test: {}'.format(a_t))
        print('\t - val : {}'.format(a_v))
        with open('{}/{}/valid.jsonl'.format(cache_dir, da), 'w') as f:
            f.write('\n'.join([json.dumps(x) for x in v]))
        with open('{}/{}/test.jsonl'.format(cache_dir, da), 'w') as f:
            f.write('\n'.join([json.dumps(x) for x in t]))

