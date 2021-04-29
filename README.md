# Analogy Tools 
This repository is aimed to collect resources for word analogy and lexical relation research:   
- Analogy Test Dataset: [description](#analogy-test-dataset), [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.tar.gz)
- Lexical Relation Dataset: [description](#lexical-relation-dataset), [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.tar.gz)
- RELATIVE embedding model:
    - [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) based model. [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.bin.tar.gz)
    - [wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html) based model. [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.bin.tar.gz)
    - [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) based model. [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.bin.tar.gz)

Available aliases of released resource by third party:
- `GoogleNews-vectors-negative300`: [Google's released word2vec model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
[link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/GoogleNews-vectors-negative300.bin.gz)
- `BATS_3.0`: Original [BATS dataset](https://vecto.space/projects/BATS/). [link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/BATS_3.0.zip)
- `glove.840B.300d.gensim`: Largest GloVe embedding model shared by [Stanford](https://nlp.stanford.edu/projects/glove/), converted to gensim format.
[link](https://drive.google.com/file/d/1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN/view?usp=sharing)

## Analogy Test Dataset
Five different datasets for word analogy. Each contains jsonline files for validation and test, in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "pmi_pred": 1,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
``` 
where `stem` is the query word pair, `choice` has word pair candidates, `pmi_pred` is statistical baseline
and `answer` indicates the index of correct candidate. Data statistics are summarized as below.

| Dataset | Size (valid/test) | Num of choice | Num of relation group |
|---------|---------:|--------------:|----------------------:|
| sat     | 37/337   | 5             | 2                     |
| u2      | 24/228   | 5,4,3         | 9                     |
| u4      | 48/432   | 5,4,3         | 5                     |
| google  | 50/500   | 4             | 2                     |
| bats    | 199/1799 | 4             | 3                     |

All data is lowercased except Google dataset.

<details><summary>Leader Board</summary>

To get word embedding baseline, 
```shell script
pytho analogy_test.py
```
</details>


## Lexical Relation Dataset
Five different datasets for lexical relation classification used in [SphereRE](https://www.aclweb.org/anthology/P19-1169/).
This contains `BLESS`, `CogALexV`, `EVALution`, `K&H+N`, `ROOT09` and each of them has `test.tsv` and `train.tsv`.
Each tsv file consists of lines which describe the relation type given word A and B. 
```
A   B   relation_type
```

For more detailed discussion, please take a look the SphereRE paper.

To get word embedding baseline, 
```shell script
pytho lexical_relation.py
```
 
## Pretrained Relation Embedding Model
[RELATIVE embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) models that trained on 
[common-word-pair](#common-word-pairs) are available:

- [*relative (word2vec)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.bin.tar.gz)
- [*relative (fasttext)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.bin.tar.gz)
- [*relative (glove)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.bin.tar.gz)
- [*relative_truecase (word2vec)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.truecase.bin.tar.gz)
- [*relative_truecase (fasttext)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.truecase.bin.tar.gz)
- [*relative_truecase (glove)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.truecase.bin.tar.gz)
- [*relative_concat (word2vec)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.concat.bin.tar.gz)
- [*relative_concat (fasttext)*](https://drive.google.com/u/0/uc?id=1EH0oywBo8OaNExyc5XTGIFhLvf8mZiBz&export=download)
- [*relative_concat (glove)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.concat.bin.tar.gz)
- [*relative_truecase_concat (word2vec)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.truecase.concat.bin.tar.gz)
- [*relative_truecase_concat (fasttext)*](https://drive.google.com/u/0/uc?id=1iUuCYM_UJ6FHI5yxg5UIGkXN4qqU5S3G&export=download)
- [*relative_truecase_concat (glove)*](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.truecase.concat.bin.tar.gz)


Models with `{}_concat` means the relative vector is concatenated on top of the underlying word embedding's difference, and
`{}_truecase` means the wikidump is converted into truecase by truecaser.
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

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.
