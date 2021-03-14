# Analogy Dataset
Dataset/model checlpoints for relational knowledge probing:
- Analogy test set:
    - SAT: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/sat.zip)
    - U2: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u2.zip)
    - U2: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u4.zip)
    - Google: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/google.zip)
    - BATS: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/bats.zip)
- Word pair list: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz)
- RELATIVE embedding model: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz)

## Data description
### Analogy test set
A jsonline file in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
``` 
where `stem` is the query word pair, `choice` has word pair candidates, and `answer` indicates the index of correct candidate.

### Word pair list
A json file of word (head) and its corresponding word (tail), based on PMI over the wikipedia.
This is fetched from the pretrained RELATIVE embedding released by the [official repo](https://github.com/pedrada88/relative).
- ***script to reproduce the data***: [`get_pair_vocab.py`](./get_pair_vocab.py)

### RELATIVE embedding model
We release the [RELATIVE embedding model](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) trained on 
[the common-crawl-pretrained Fasttext model released from Facebook](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip).
As the model vocabulary, we use all the pair from the above analogy test set as well as the word pair list.
It's formatted to be used in gensim:
```python
from gensim.models import KeyedVectors
relative_model = KeyedVectors.load_word2vec_format('./relative_init_vectors.bin', binary=True)
```
- ***script to train RELATIVE model***: [`get_relative_embedding.py`](./get_relative_embedding.py)

## Test Analogy 
Quick experiment to compare our RELATIVE model with the underlying FastText model.
```
{"model": "fasttext", "data": "sat", "oov_test": 0, "accuracy_test": 0.4629080118694362, "oov_valid": 0, "accuracy_valid": 0.5405405405405406}
{"model": "relative", "data": "sat", "oov_test": 252, "accuracy_test": 0.43026706231454004, "oov_valid": 22, "accuracy_valid": 0.32432432432432434}
{"model": "fasttext", "data": "u2", "oov_test": 0, "accuracy_test": 0.3815789473684211, "oov_valid": 0, "accuracy_valid": 0.2916666666666667}
{"model": "relative", "data": "u2", "oov_test": 172, "accuracy_test": 0.40789473684210525, "oov_valid": 17, "accuracy_valid": 0.25}
{"model": "fasttext", "data": "u4", "oov_test": 0, "accuracy_test": 0.38425925925925924, "oov_valid": 0, "accuracy_valid": 0.3958333333333333}
{"model": "relative", "data": "u4", "oov_test": 304, "accuracy_test": 0.3888888888888889, "oov_valid": 38, "accuracy_valid": 0.3541666666666667}
{"model": "fasttext", "data": "google", "oov_test": 0, "accuracy_test": 0.948, "oov_valid": 0, "accuracy_valid": 0.94}
{"model": "relative", "data": "google", "oov_test": 479, "accuracy_test": 0.938, "oov_valid": 47, "accuracy_valid": 0.94}
{"model": "fasttext", "data": "bats", "oov_test": 0, "accuracy_test": 0.7065036131183992, "oov_valid": 0, "accuracy_valid": 0.7336683417085427}
{"model": "relative", "data": "bats", "oov_test": 1446, "accuracy_test": 0.7242912729294052, "oov_valid": 156, "accuracy_valid": 0.7386934673366834}
```

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.