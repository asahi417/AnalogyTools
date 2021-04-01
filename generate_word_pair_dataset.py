""" Generate common word pair dataset by concatenating the two sources:
(i) the vocabulary of RELATIVE vector trained on wikipedia
"""
import pickle
import logging
import truecase
from util import get_pair_relative

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
all_vocab = get_pair_relative()
all_vocab = [v.split('__') for v in list(set(['__'.join(v) for v in all_vocab]))]
logging.info('{} pairs'.format(len(all_vocab)))

with open('./common_word_pairs.pkl', "wb") as fp:
    pickle.dump(all_vocab, fp)

logging.info('truecasing')


def tc(string):
    string = truecase.get_true_case('A ' + string)
    return string[2:]


all_vocab_tc = [[tc(v_) for v_ in v] for v in all_vocab]
with open('./common_word_pairs_truecase.pkl', "wb") as fp:
    pickle.dump(all_vocab_tc, fp)
