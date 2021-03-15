""" Convert fasttext embedding to relational embedding format:
The embedding for word A and B is computed as, ft[A] - ft[B] and it's stored by the index of
'{}__{}'.format(A.lower(), B.lowe()). Eg) Dft["paris_france"] = ft["Paris"] - ft["France"] where
Dft is the new model and ft is the underlying fasttext model.
"""
import logging
import os
import argparse
from itertools import groupby
from typing import Dict
from tqdm import tqdm

from gensim.models import fasttext
from gensim.models import KeyedVectors
from util import open_compressed_file
from get_relative_embedding import get_pair_analogy, get_pair_relative

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Anchor word embedding model
URL_WORD_EMBEDDING = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
PATH_WORD_EMBEDDING = './cache/crawl-300d-2M-subword.bin'
if not os.path.exists(PATH_WORD_EMBEDDING):
    logging.info('downloading fasttext model')
    open_compressed_file(url=URL_WORD_EMBEDDING, cache_dir='./cache')


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
                _tail = tail.replace(' ', '_')
                _head = head.replace(' ', '_')
                # index is created in lower case although the embedding is based on case-sensitive
                body = '__'.join([_head.lower(), _tail.lower()])
                for v in relative_embedding:
                    body += ' ' + str(v)
                body += "\n"
                txt_file.write(body)
                line_count += 1

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
    logging.info("compute difference of embedding")

    cache = opt.output.replace('.bin', '.txt')
    if not os.path.exists(cache):
        pair_vocab = get_pair_relative(uncased=False) + get_pair_analogy(uncased=False)
        grouper = groupby(pair_vocab, key=lambda x: x[0])
        pair_vocab_dict = {k: list(set(list(map(lambda x: x[1], g)))) for k, g in grouper}
        get_diff_vec(cache, pair_vocab_dict)

    logging.info("producing binary file")
    model = KeyedVectors.load_word2vec_format(cache)
    model.wv.save_word2vec_format(opt.output, binary=True)
    logging.info("new embeddings are available at {}".format(opt.output))
