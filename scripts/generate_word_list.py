import pickle
from util import get_word_embedding_model

model = get_word_embedding_model('glove')
vocab_glove = set(model.vocab.keys())
model = get_word_embedding_model('w2v')
vocab_w2v = set(model.vocab.keys())
del model
vocab = vocab_glove.intersection(vocab_w2v)
vocab = sorted(list(vocab))

with open('./common_word.pkl', "wb") as fp:
    pickle.dump(vocab, fp)
