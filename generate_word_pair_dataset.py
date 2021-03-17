""" Generate common word pair dataset by concatenating the two sources:
(i) the vocabulary of RELATIVE vector trained on wikipedia
(ii) the word pair from five analogy test dataset
"""
import pickle
import logging
from util import get_pair_analogy, get_pair_relative

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
all_vocab = get_pair_analogy() + get_pair_relative()
all_vocab = [v.split('__') for v in list(set(['__'.join(v) for v in all_vocab]))]

with open('./common_word_pairs.pkl', "wb") as fp:
    pickle.dump(all_vocab, fp)
