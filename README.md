# Analogy Tools 
This repository is aimed to collect resources for word analogy and lexical relation research.
- Analogy Test Dataset: [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.tar.gz)
- Lexical Relation Dataset: [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.tar.gz)
- RELATIVE embedding model:
    - [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.bin.tar.gz)
    - [wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.bin.tar.gz)
    - [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.bin.tar.gz)
- Word embedding model:
    - Largest GloVe embedding model shared by [Stanford](https://nlp.stanford.edu/projects/glove/), converted to gensim format. [***link***](https://drive.google.com/file/d/1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN/view?usp=sharing)

Available aliases of released resource by third party:
- [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit): [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/GoogleNews-vectors-negative300.bin.gz)
- [BATS_3.0](https://vecto.space/projects/BATS/): [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/BATS_3.0.zip)

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

<details><summary> Leader Board</summary>

Here we report baseline with word embedding model. Complete result can be found [here](./results/analogy_test.csv), 
and please refer [our script](analogy_test.py) for more detail experimental setting.

- ***BATS***

| model    | accuracy (val) | accuracy (test) | accuracy (full) |
|----------|----------------|-----------------|-----------------|
| fasttext |         75.88% |          74.21% |          74.37% |
| glove    |         70.85% |          67.32% |          67.67% |
| w2v      |         65.33% |          63.15% |          63.36% |
| PMI      |         35.18% |          42.75% |          41.99% |

- ***Google***

| model    | accuracy (val) | accuracy (test) | accuracy (full) |
|----------|----------------|-----------------|-----------------|
| fasttext |         94.00% |          97.40% |          97.09% |
| glove    |         92.00% |          96.00% |          95.64% |
| w2v      |         92.00% |          93.40% |          93.27% |
| PMI      |         60.00% |          57.40% |          57.64% |

- ***SAT***

| model    | accuracy (val) | accuracy (test) | accuracy (full) |
|----------|----------------|-----------------|-----------------|
| fasttext |         67.57% |          47.77% |          49.73% |
| glove    |         59.46% |          47.77% |          48.93% |
| w2v      |         56.76% |          44.21% |          45.45% |
| PMI      |         24.32% |          23.15% |          23.26% |

- ***U2***

| model    | accuracy (val) | accuracy (test) | accuracy (full) |
|----------|----------------|-----------------|-----------------|
| fasttext |         50.00% |          42.98% |          43.65% |
| glove    |         41.67% |          44.74% |          44.44% |
| w2v      |         37.50% |          41.23% |          40.87% |
| PMI      |         29.17% |          32.89% |          32.54% |

- ***U4***

| model    | accuracy (val) | accuracy (test) | accuracy (full) |
|----------|----------------|-----------------|-----------------|
| fasttext |         47.92% |          37.04% |          38.13% |
| glove    |         52.08% |          35.19% |          36.88% |
| w2v      |         37.50% |          36.57% |          36.67% |
| PMI      |         47.92% |          39.12% |          40.00% |

To get word embedding baseline, 
```shell script
python analogy_test.py
```
When the model suffers out-of-vocabulary error, we use PMI prediction, `pmi_pred` in each entry, to ensure the baseline can
be compared with other methods to cover all the data points.   

</details>


## Lexical Relation Dataset
Five different datasets for lexical relation classification used in [SphereRE](https://www.aclweb.org/anthology/P19-1169/).
This contains `BLESS`, `CogALexV`, `EVALution`, `K&H+N`, `ROOT09` and each of them has `test.tsv` and `train.tsv`.
Each tsv file consists of lines which describe the relation type given word A and B. 
```
A   B   relation_type
```
For more detailed discussion, please take a look the [SphereRE](https://www.aclweb.org/anthology/P19-1169/) paper.

<details><summary>Leader Board</summary>

Here we report baseline with word embedding model. Complete result can be found [here](./results/lexical_relation.csv), 
and please refer [our script](lexical_relation.py) for more detail experimental setting.

- ***BLESS*** 

| model    | accuracy | f1_macro | f1_micro |
|----------|----------|----------|----------|
| fasttext |   92.87% |   92.34% |   92.87% |
| glove    |   93.22% |   92.63% |   93.22% |
| w2v      |   92.30% |   91.84% |   92.30% |

- ***CogALexV***

| model    | accuracy | f1_macro | f1_micro |
|----------|----------|----------|----------|
| fasttext |   78.10% |   52.06% |   78.10% |
| glove    |   79.23% |   52.82% |   79.23% |
| w2v      |   77.37% |   49.10% |   77.37% |

- ***EVALution***

| model    | accuracy | f1_macro | f1_micro |
|----------|----------|----------|----------|
| fasttext |   57.04% |   55.49% |   57.04% |
| glove    |   58.34% |   57.59% |   58.34% |
| w2v      |   56.66% |   55.61% |   56.66% |

- ***K&H+N***

| model    | accuracy | f1_macro | f1_micro |
|----------|----------|----------|----------|
| fasttext |   93.97% |   88.46% |   93.97% |
| glove    |   95.26% |   90.15% |   95.26% |
| w2v      |   90.81% |   84.76% |   90.81% |

- ***ROOT09***

| model    | accuracy | f1_macro | f1_micro |
|----------|----------|----------|----------|
| fasttext |   88.84% |   88.67% |   88.84% |
| glove    |   89.22% |   88.64% |   89.22% |
| w2v      |   87.15% |   86.65% |   87.15% |

To get word embedding baseline, 
```shell script
python lexical_relation.py
```
When the model suffers out-of-vocabulary error in evaluation, we use the most frequent label in training data, to ensure the baseline can
be compared with other methods to cover all the data points.   

</details>
 

## RELATIVE Embedding
[RELATIVE embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) models extract relation embedding from the anchor word embedding model 
by aggregating coocurring word in between the word pairs given a large corpus. We present three models each corresponds to major pretrained public word embedding model,
[GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), [wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html), and [glove.840B.300d](https://nlp.stanford.edu/projects/glove/).
The binary files are supported by gensim:
```python
In [1] from gensim.models import KeyedVectors
In [2] relative_model = KeyedVectors.load_word2vec_format('relative_init.glove.bin', binary=True)
In [3] relative_model['paris__france']
Out[4] 
array([-1.16878878e-02, ... 7.91083463e-03], dtype=float32)  # 300 dim array
```
Note that words are joined by `__` and all the vocabulary is uncased. Multiple token should be combined by `_` such as 
`new_york__tokyo` for the relation across New York and Tokyo.

To reproduce relative model, run the following code.

```shell script
python calculate_relative_embedding.py
```

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.

