# Analogy Data and Relative Embedding 
Release five analogy datasets and relative embedding models trained to cover the relation pairs.

## Analogy Test Dataset
Following analogy dataset is available (click to download the data):
[SAT](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/sat.zip), 
[U2](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u2.zip),
[U4](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u4.zip),
[Goolgle](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/google.zip),
[BATS](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/bats.zip).

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

## Common Word Pairs
We provide a json file of a head word and its corresponding tail words, based on PMI over the lower cased Wikipedia
[here](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz).
This is fetched from the pretrained RELATIVE embedding released by the [official repo](https://github.com/pedrada88/relative) and 
is supposed to be used as a corpus to train a relation embedding model.

- ***script to reproduce the data***: [`get_pair_vocab.py`](./get_pair_vocab.py)

## RELATIVE embedding model
We release the [RELATIVE embedding model](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) trained with 
[the common-crawl-pretrained Fasttext model released from Facebook](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip)
(*relative_init_vectors*) over the [common-word-pair](#common-word-pairs) and all the word pairs from [analogy test dataset](#analogy-test-dataset).
As a comparison, we also provide an embedding model with the same format but converted from fasttext trained on common-crawl,
where we take the difference in between each word pair and regard it as a relative vector (*fasttext_diff_vectors*).
Finally, we simply concat *relative_init_vectors* and *fasttext_diff_vectors* that is referred as
*concat_relative_fasttext_vectors*.

- [*relative_init_vectors*](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_init_vectors.bin.tar.gz)
- [*fasttext_diff_vectors*](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/fasttext_diff_vectors.bin.tar.gz)
- [*concat_relative_fasttext_vectors*](https://drive.google.com/u/0/uc?id=1CkdsxEl21TUiBmLS6uq55tH6SiHvWGDn&export=download)

As the model vocabulary, we use all the pair from the above analogy test set as well as the word pair list.
It's formatted to be used in gensim:
```python
from gensim.models import KeyedVectors
relative_model = KeyedVectors.load_word2vec_format('relative_init_vectors.bin', binary=True)
relative_model['paris__france']
```
Note that words are joined by `__` and all the vocabulary is uncased. Multiple token should be combined by `_` such as 
`new_york__tokyo` for the relation in between New York and Tokyo. The *fasttext_diff_vectors* model also relies on lowercase,
but at the model construction, we use case-sensitive vocabulary, i.e.
```
new_embedding["new_york__tokyo"] = fasttext_model["New York"] - fasttext_model["Tokyo"]
```

- ***script to train relative_init_vectors***: [`get_relative_init.py`](get_relative_init.py)
- ***script to produce fasttext_diff_vectors***: [`get_fasttext_diff.py`](get_fasttext_diff.py)

## Analogy Result 
Benchmark of the analogy dataset with our relative embeddings. Fasttext can handle any words so when other model
has out-of-vocabulary (OOV), we simply use the fasttext's prediction instead.
We export fasttext prediction as a default baseline [here](./fasttext_prediction.json).

| data   | model                    | oov_test | accuracy_test | oov_valid | accuracy_valid | accuracy |
|--------|--------------------------|----------|---------------|-----------|----------------|----------|
| bats   | concat_relative_fasttext | 399      | 0.685937      | 50        | 0.738693       | 0.691191 |
| bats   | fasttext_diff            | 0        | 0.714842      | 0         | 0.743719       | 0.717718 |
| bats   | relative_init            | 399      | 0.517510      | 50        | 0.537688       | 0.519520 |
| google | concat_relative_fasttext | 65       | 0.870000      | 6         | 0.840000       | 0.867273 |
| google | fasttext_diff            | 0        | 0.948000      | 0         | 0.940000       | 0.947273 |
| google | relative_init            | 65       | 0.654000      | 6         | 0.640000       | 0.652727 |
| sat    | concat_relative_fasttext | 88       | 0.376855      | 4         | 0.405405       | 0.379679 |
| sat    | fasttext_diff            | 0        | 0.462908      | 0         | 0.540541       | 0.470588 |
| sat    | relative_init            | 88       | 0.311573      | 4         | 0.243243       | 0.304813 |
| u2     | concat_relative_fasttext | 49       | 0.372807      | 5         | 0.458333       | 0.380952 |
| u2     | fasttext_diff            | 0        | 0.381579      | 0         | 0.291667       | 0.373016 |
| u2     | relative_init            | 49       | 0.342105      | 5         | 0.250000       | 0.333333 |
| u4     | concat_relative_fasttext | 83       | 0.335648      | 8         | 0.437500       | 0.345833 |
| u4     | fasttext_diff            | 0        | 0.384259      | 0         | 0.395833       | 0.385417 |
| u4     | relative_init            | 83       | 0.293981      | 8         | 0.333333       | 0.297917 |

- ***script to reproduce the result***: [`test_analogy.py`](./test_analogy.py)

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.
For the fasttext model, please refer [the facebook release](https://fasttext.cc/docs/en/english-vectors.html).
