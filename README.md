# Analogy Data and Relative Embedding 
Analogy dataset and relative embedding model trained to cover the relation.

## Analogy test set
Following analogy dataset is available:
- SAT: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/sat.zip)
- U2: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u2.zip)
- U2: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u4.zip)
- Google: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/google.zip)
- BATS: [file](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/bats.zip)

Each contains jsonline files for validation and test, in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
``` 
where `stem` is the query word pair, `choice` has word pair candidates, and `answer` indicates the index of correct candidate.

## Word pair list
We provide a json file of word (head) and its corresponding word (tail), based on PMI over the lower cased Wikipedia
[here](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz).
This is fetched from the pretrained RELATIVE embedding released by the [official repo](https://github.com/pedrada88/relative).
- ***script to reproduce the data***: [`get_pair_vocab.py`](./get_pair_vocab.py)

## RELATIVE embedding model
We release the [RELATIVE embedding model](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) trained on 
[the common-crawl-pretrained Fasttext model released from Facebook](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip)
(*relative_init_vectors*).
As a comparison, we also provide an embedding model with the same format but converted from fasttext trained on common-crawl,
where we take the difference in between each word pair and regard it as a relative vector (*fasttext_diff_vectors*).

- [*relative_init_vectors*](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz)
- [*fasttext_diff_vectors*](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz)

As the model vocabulary, we use all the pair from the above analogy test set as well as the word pair list.
It's formatted to be used in gensim:
```python
from gensim.models import KeyedVectors
relative_model = KeyedVectors.load_word2vec_format('./relative_init_vectors.bin', binary=True)
relative_model['paris__france']
```
Note that words are joined by `__` and all the vocabulary is uncased. Multiple token should be combined by `_` such as 
`new_york__tokyo` for the relation in between New York and Tokyo. 

- ***script to train relative_init_vectors***: [`get_relative_embedding.py`](./get_relative_embedding.py)
- ***script to produce fasttext_diff_vectors***: [`convert_fasttext.py`](./convert_fasttext.py)

## Analogy Result 
Quick experiment to compare our RELATIVE model with the underlying FastText model.

| model    | data   | oov_test | accuracy_test | oov_valid | accuracy_valid |
|----------|--------|----------|---------------|-----------|----------------|
| fasttext | sat    | 0        | 0.462908      | 0         | 0.540541       |
| fasttext | u2     | 0        | 0.381579      | 0         | 0.291667       |
| fasttext | u4     | 0        | 0.384259      | 0         | 0.395833       |
| fasttext | google | 0        | 0.948000      | 0         | 0.940000       |
| fasttext | bats   | 0        | 0.706504      | 0         | 0.733668       |
| relative | sat    | 252      | 0.430267      | 22        | 0.324324       |
| relative | u2     | 172      | 0.407895      | 17        | 0.250000       |
| relative | u4     | 304      | 0.388889      | 38        | 0.354167       |
| relative | google | 479      | 0.938000      | 47        | 0.940000       |
| relative | bats   | 1446     | 0.724291      | 156       | 0.738693       |

- ***script to reproduce the result***: [`test_analogy.py`](./test_analogy.py)

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.
For the fasttext model, please refer [the facebook release](https://fasttext.cc/docs/en/english-vectors.html).
