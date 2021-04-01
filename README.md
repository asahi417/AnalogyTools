# Analogy Data and Relative Embedding 
Five analogy datasets and relative embedding models trained on Wikipedia dump.

## Analogy Test Dataset
Following analogy dataset is available (click to download the data):
[SAT](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/sat.zip), 
[U2](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/u2.zip),
[U4](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/u4.zip),
[Goolgle](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/google.zip),
[BATS](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/bats.zip).

Each contains jsonline files for validation and test, in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
``` 
where `stem` is the query word pair, `choice` has word pair candidates, and `answer` indicates the index of correct candidate. Data statistics are summarized as below.

| Dataset | Size (valid/test) | Num of choice | Num of relation group |
|---------|---------:|--------------:|----------------------:|
| sat     | 37/337   | 5             | 2                     |
| u2      | 24/228   | 5,4,3         | 9                     |
| u4      | 48/432   | 5,4,3         | 5                     |
| google  | 50/500   | 4             | 2                     |
| bats    | 199/1799 | 4             | 3                     |


## Common Word Pairs
Common word pair dataset is a dataset consisting of a pair of head and tail word ([link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl.tar.gz)).
The dataset is built on top of lowercased wikipedia dump. 

```python
In [1] import pickle
In [2] 
def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)
In [3] data = load_pickle('common_word_pairs.pkl')
In [4] data[:2]
Out[4] [['prosperity', 'century'], ['haileybury', 'imperial']]
```

- ***script to reproduce the data***: [`generate_word_pair_dataset.py`](generate_word_pair_dataset.py)

## Pretrained Relation Embedding Model
Following [RELATIVE embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) models that trained on 
[common-word-pair](#common-word-pairs) are available:

- [*relative_init_vectors (word2vec)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.bin.tar.gz)
- [*relative_init_vectors (fasttext)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.bin.tar.gz)
- [*relative_init_vectors (glove)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.bin.tar.gz)

The binary file is supposed to be used via gensim:
```python
In [1] from gensim.models import KeyedVectors
In [2] relative_model = KeyedVectors.load_word2vec_format('relative_init_vectors.bin', binary=True)
In [3] relative_model['paris__france']
Out[4] 
array([-1.16878878e-02, ... 7.91083463e-03], dtype=float32)  # 300 dim array
```
Note that words are joined by `__` and all the vocabulary is uncased. Multiple token should be combined by `_` such as 
`new_york__tokyo` for the relation across New York and Tokyo.

- ***script to train relative_init_vectors***: [`get_relative_init.py`](get_relative_init.py)

## Analogy Test Baseline 
Here we show baselines of the analogy dataset with our relative embeddings. Fasttext can handle any words so when other model
has out-of-vocabulary (OOV), we simply use the fasttext's prediction instead.

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

The prediction from each model is exported at [here](./predictions). 

- ***script to reproduce the result***: [`test_analogy.py`](analogy_test.py)

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.
For the fasttext model, please refer [the facebook release](https://fasttext.cc/docs/en/english-vectors.html).
