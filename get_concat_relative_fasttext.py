""" generate `concat_relative_fasttext` embedding model """
import os
import logging
from tqdm import tqdm
from gensim.models import KeyedVectors
from util import get_embedding_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

path_concat_embedding = './cache/concat_relative_fasttext_vectors.bin'

relative = get_embedding_model('relative_init')
fasttext = get_embedding_model('fasttext_diff')

os.makedirs(os.path.dirname(path_concat_embedding), exist_ok=True)
with open(path_concat_embedding.replace('.bin', '.txt'), 'w') as f:
    f.write(str(len(relative.vocab)) + " " + str(relative.vector_size + fasttext.vector_size) + "\n")
    for v in tqdm(relative.vocab):
        new_vector = list(relative[v]) + list(fasttext[v])
        f.write(v + ' ' + ' '.join([str(i) for i in new_vector]) + "\n")

logging.info("producing binary file")
model = KeyedVectors.load_word2vec_format(path_concat_embedding.replace('.bin', '.txt'))
model.wv.save_word2vec_format(path_concat_embedding, binary=True)

