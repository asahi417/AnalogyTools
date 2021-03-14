""" Convert fasttext embedding to relational embedding format """
import logging
import os
import json
import pickle
import argparse
from itertools import chain, groupby
from typing import Dict
from tqdm import tqdm

from gensim.models import fasttext
from gensim.models import KeyedVectors
from util import open_compressed_file

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Anchor word embedding model
URL_WORD_EMBEDDING = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
PATH_WORD_EMBEDDING = './cache/crawl-300d-2M-subword.bin'
if not os.path.exists(PATH_WORD_EMBEDDING):
    logging.info('downloading fasttext model')
    open_compressed_file(url=URL_WORD_EMBEDDING, cache_dir='./cache')


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


def get_diff_vec(output_path: str, relation_pairs: Dict):
    """ Get difference in between relation word embeddings """
    logging.info("loading embeddings")
    word_embedding_model = fasttext.load_facebook_model(PATH_WORD_EMBEDDING)
    line_count = 0
    logging.info("convert embeddings")
    with open(output_path + '.tmp', 'w', encoding='utf-8') as txt_file:
        for head, tails in tqdm(relation_pairs.items()):
            for tail in tails:
                relative_embedding = word_embedding_model[head] - word_embedding_model[tail]
                body = '__'.join([head, tail])
                # txt_file.write()
                # for v in relative_embedding.tolist():
                for v in relative_embedding:
                    # if abs(v) < 1e-4:
                    #     txt_file.write(' 0')
                    # else:
                    #     txt_file.write(' ' + str(v))
                    body += ' ' + str(v)
                    # txt_file.write(' ' + str(v))
                body += "\n"
                print(body)
                print(len(body.rsplit()), body.rsplit())
                print(len(body.split(' ')), body.split(' '))
                txt_file.write(body)
                line_count += 1
                input()
    logging.info("reformat file to add header")
    logging.info("\t * {} lines, {} dim".format(line_count, word_embedding_model.vector_size))
    with open(output_path, 'w') as f_out:
        f_out.write(str(line_count) + " " + str(word_embedding_model.vector_size) + "\n")
        with open(output_path + '.tmp', 'r') as f_cache:
            for line in f_cache:
                f_out.write(line)


def get_options():
    parser = argparse.ArgumentParser(description='simplified RELATIVE embedding training')
    parser.add_argument('-o', '--output', help='Output file path to store relation vectors',
                        type=str, default="./cache/fasttext_diff_vectors.bin")
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    assert opt.output.endswith('.bin')
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

    logging.info("compute difference of embedding")

    cache = opt.output.replace('.bin', '.txt')
    if not os.path.exists(cache):
        get_diff_vec(cache, pair_vocab_dict)

    logging.info("producing binary file")
    model = KeyedVectors.load_word2vec_format(cache)
    model.wv.save_word2vec_format(opt.output, binary=True)
    logging.info("new embeddings are available at {}".format(opt.output))
