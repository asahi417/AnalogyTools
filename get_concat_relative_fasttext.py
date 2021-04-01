""" generate `concat_relative_fasttext` embedding model """
import argparse
import os
import logging
from tqdm import tqdm
from gensim.models import KeyedVectors
from util import get_word_embedding_model, get_relative_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='concat RELATIVE and word embedding')
    parser.add_argument('-o', '--output-dir', help='Output file path to store relation vectors',
                        type=str, default="./cache")
    parser.add_argument('--model-word', help='word embedding model', type=str, default="fasttext")
    parser.add_argument('--model-relative', help='relative embedding model', type=str, default="fasttext")

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()

    model_relative = get_relative_embedding_model(opt.model_relative)
    model_word = get_word_embedding_model(opt.model_word)

    os.makedirs(opt.output_dir, exist_ok=True)
    model = '{}/'.format(opt.output_dir, )
    with open(path_concat_embedding.replace('.bin', '.txt'), 'w') as f:
        f.write(str(len(relative.vocab)) + " " + str(relative.vector_size + fasttext.vector_size) + "\n")
        for v in tqdm(relative.vocab):
            new_vector = list(relative[v]) + list(fasttext[v])
            f.write(v + ' ' + ' '.join([str(i) for i in new_vector]) + "\n")

    logging.info("producing binary file")
    model = KeyedVectors.load_word2vec_format(path_concat_embedding.replace('.bin', '.txt'))
    model.wv.save_word2vec_format(path_concat_embedding, binary=True)

