""" Simplified script to get RELATIVE vector for fixed pair word dataset on Wikipedia dump with FastText """
import logging
import os
import json
import pickle
import argparse
from itertools import chain, groupby
from typing import Dict
from tqdm import tqdm


from gensim.models import fasttext
from util import open_compressed_file

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Anchor word embedding model
URL_WORD_EMBEDDING = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
PATH_WORD_EMBEDDING = './cache/crawl-300d-2M-subword.bin'
if not os.path.exists(PATH_WORD_EMBEDDING):
    logging.info('downloading fasttext model')
    open_compressed_file(url=URL_WORD_EMBEDDING, cache_dir='./cache')

# Corpus
URL_CORPUS = 'https://drive.google.com/u/0/uc?id=17EBy4GD4tXl9G4NTjuIuG5ET7wfG4-xa&export=download'
PATH_CORPUS = './cache/wikipedia_en_preprocessed.txt'
CORPUS_LINE_LEN = 104000000
if not os.path.exists(PATH_CORPUS):
    logging.info('downloading wikidump')
    open_compressed_file(url=URL_CORPUS, cache_dir='./cache', filename='wikipedia_en_preprocessed.zip', gdrive=True)

# Stopwords
with open('./stopwords_en.txt', 'r') as f:
    STOPWORD_LIST = list(set(list(filter(len, f.read().split('\n')))))


def get_pair_relative():
    """ Get the list of word pairs in RELATIVE pretrained model """
    _path = './cache/relative_vocab.pkl'
    if not os.path.exists(_path):
        url = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz'
        open_compressed_file(url=url, cache_dir='./cache')
    with open(_path, "rb") as fp:
        return pickle.load(fp)


def get_pair_analogy():
    """ Get the list of word pairs in analogy dataset """
    data = dict(
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


def get_word_from_corpus(minimum_frequency: int, word_vocabulary_size: int = None):
    """ Get word distribution over corpus """
    dict_freq = {}
    bar = tqdm(total=CORPUS_LINE_LEN)
    with open(PATH_CORPUS, 'r', encoding='utf-8') as corpus_file:
        for _line in corpus_file:
            bar.update()
            l_split = _line.strip().split(" ")
            for token in l_split:
                if token in STOPWORD_LIST or "__" in token or token.isdigit():
                    continue
                if token in dict_freq:
                    dict_freq[token] += 1
                else:
                    dict_freq[token] = 1
    # frequency filter
    dict_freq = sorted(dict_freq.items(), key=lambda x: x[0])
    n = 0
    for n, (_, freq) in enumerate(dict_freq):
        if freq >= minimum_frequency:
            break

    if word_vocabulary_size is not None:
        m = min(len(dict_freq), word_vocabulary_size + n)
        dict_freq = {k: v for k, v in dict_freq[n:m]}
    else:
        dict_freq = {k: v for k, v in dict_freq[n:]}
    return list(dict_freq.keys())


def frequency_filtering(vocab, dict_pairvocab, window_size):

    def get_context(i, tokens):
        """ get context with token `i` in `tokens`, returns list of tuple (token_j, [w_1, ...])"""
        context_i_ = [(tokens[j], tokens[i + 1:j]) for j in
                      range(i + 2, min(i + 1 + window_size, len(tokens))) if tokens[j] in dict_pairvocab[tokens[i]]]
        if len(context_i_) == 0:
            return {}
        return dict([(k, list(g)[0][1]) for k, g in groupby(context_i_, key=lambda x: x[0])])

    def get_frequency(_list):
        """ return dictionary with its occurrence """
        return dict([(k, len(list(i))) for k, i in groupby(_list) if k in vocab])

    def safe_query(_dict, _key):
        if _key in _dict:
            return _dict[_key]
        else:
            return []

    context_word_dict = {}
    logging.info('start computing context word')
    bar = tqdm(total=CORPUS_LINE_LEN)
    with open(PATH_CORPUS, 'r', encoding='utf-8') as corpus_file:
        for sentence in corpus_file:
            bar.update()
            token_list = sentence.strip().split(" ")
            contexts = [(i_, token_i_, get_context(i_, token_list)) for i_, token_i_ in enumerate(token_list)
                        if token_i_ in dict_pairvocab.keys()]
            # print(contexts)
            # input()
            for i_, token_i_, context_i in contexts:
                if len(context_i) == 0:
                    continue
                if token_i_ not in context_word_dict.keys():
                    context_word_dict[token_i_] = {}
                keys = set(context_word_dict[token_i_].keys()).union(set(context_i.keys()))
                context_word_dict[token_i_] = {
                    k: safe_query(context_word_dict[token_i_], k) + safe_query(context_i, k) for k in keys}
            # print(context_word_dict)
            # input()
    logging.info('aggregating to get frequency')
    context_word_dict = {k: {k_: get_frequency(v_) for k_, v_ in v.items()} for k, v in context_word_dict.items()}

    return context_word_dict


def get_relative_init(output_path: str,
                      context_word_dict: Dict,
                      minimum_frequency_context: int):
    """ Get RELATIVE vectors """
    logging.info("loading embeddings")
    word_embedding_model = fasttext.load_facebook_model(PATH_WORD_EMBEDDING)

    with open(output_path, 'w', encoding='utf-8') as txt_file:
        for token_i, tokens_paired in context_word_dict.items():
            for token_j in tokens_paired:
                vector_pair = 0
                cont_pair = 0
                for token_co in context_word_dict[token_i][token_j]:
                    freq = context_word_dict[token_i][token_j][token_co]
                    if freq < minimum_frequency_context:
                        continue
                    token_co_vector = word_embedding_model[token_co]
                    vector_pair += (freq * token_co_vector)
                    cont_pair += 1
                if cont_pair != 0:
                    vector_pair = vector_pair/cont_pair
                    txt_file.write(','.join([token_i, token_j] + list(map(str, vector_pair.tolist()))))                    
                    txt_file.write("\n")
    logging.info("new embeddings are available at {}".format(output_path))


def get_options():
    parser = argparse.ArgumentParser(description='simplified RELATIVE embedding training')
    parser.add_argument('-o', '--output', help='Output file path to store relation vectors',
                        type=str, default="./cache/relative_init_vectors.txt")
    # The following parameters are needed if contexts are not provided
    parser.add_argument('-w', '--window-size', help='Co-occurring window size', type=int, default=10)
    parser.add_argument('--minimum-frequency-context', default=1, type=int,
                        help='Minimum frequency of words between word pair: increasing the number can speed up the'
                             'calculations and reduce memory but we would recommend keeping this number low')

    # The following parameters are needed if pair vocabulary is not provided
    parser.add_argument('--minimum-frequency', help='Minimum frequency of words', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':

    opt = get_options()
    if os.path.exists(opt.output):
        exit('found file at {}'.format(opt.output))

    os.makedirs(os.path.dirname(opt.output), exist_ok=True)

    logging.info("retrieve pair and word vocabulary (dictionary)")
    cache = '{}/pair_vocab.json'.format(os.path.dirname(opt.output))
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            pair_vocab_dict = json.load(f)
    else:
        pair_relative = get_pair_relative()
        pair_analogy = get_pair_analogy()
        pair_vocab = pair_analogy + pair_relative
        grouper = groupby(pair_vocab, key=lambda x: x[0])
        pair_vocab_dict = {k: list(set(list(map(lambda x: x[1], g)))) for k, g in grouper}
        with open(cache, 'w') as f:
            json.dump(pair_vocab_dict, f)

    logging.info("extracting contexts(this can take a few hours depending on the size of the corpus)")
    logging.info("\t * loading word frequency dictionary")
    path = '{}/vocab.pkl'.format(os.path.dirname(opt.output))
    if os.path.exists(path):
        with open(path, 'rb') as fb:
            vocab_ = pickle.load(fb)
    else:
        vocab_ = get_word_from_corpus(minimum_frequency=opt.minimum_frequency)
        with open(path, 'wb') as fb:
            pickle.dump(vocab_, fb)

    logging.info("\t * filtering corpus by frequency")
    cache = '{}/pairs_context.json'.format(os.path.dirname(opt.output))
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            pairs_context = json.load(f)
    else:
        pairs_context = frequency_filtering(vocab_, pair_vocab_dict, opt.window_size)
        with open(cache, 'w') as f:
            json.dump(pairs_context, f)

    logging.info("computing relative-init vectors")
    get_relative_init(
        output_path=opt.output,
        context_word_dict=pairs_context,
        minimum_frequency_context=opt.minimum_frequency_context)



