# Analogy Tools 
This repository is a collection of resources for word analogy and lexical relation research.
- Analogy Test Dataset: [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset_no_prediction.tar.gz)
- Lexical Relation Dataset: [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.tar.gz)
- RELATIVE embedding model:
    - [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.w2v.bin.tar.gz)
    - [wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.fasttext.bin.tar.gz)
    - [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) based model. [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relative_init.glove.bin.tar.gz)
- Word embedding model:
    - Largest GloVe embedding model shared by [Stanford](https://nlp.stanford.edu/projects/glove/), converted to gensim format. [***link***](https://drive.google.com/file/d/1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN/view?usp=sharing)

Aliases of released resource by third party:
- [GoogleNews-vectors-negative300](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit): [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/GoogleNews-vectors-negative300.bin.gz)
- [BATS_3.0](https://vecto.space/projects/BATS/): [***link***](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/BATS_3.0.zip)

The LICENSE of all the resources are under [CC-BY-NC-4.0](./LICENSE). Thus, they are freely available for academic purpose or individual research, but restricted for commecial use.

## Analogy Test Dataset
We release the five different word analogy dataset in the following links: 
- [dataset](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset_no_prediction.tar.gz)
- [dataset with baseline prediction](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.tar.gz)

The first file contains the dataset while second file has model prediction from PMI and some word embedding models. Each contains jsonline files for validation and test, in which each line consists of following dictionary,
```
{"stem": ["raphael", "painter"],
 "answer": 2,
 "choice": [["andersen", "plato"],
            ["reading", "berkshire"],
            ["marx", "philosopher"],
            ["tolstoi", "edison"]]}
``` 
where `stem` is the query word pair, `choice` has word pair candidates, 
and `answer` indicates the index of correct candidate. Data statistics are summarized as below.

| Dataset | Size (valid/test) | Num of choice | Num of relation group |
|---------|---------:|--------------:|----------------------:|
| sat     | 37/337   | 5             | 2                     |
| u2      | 24/228   | 5,4,3         | 9                     |
| u4      | 48/432   | 5,4,3         | 5                     |
| google  | 50/500   | 4             | 2                     |
| bats    | 199/1799 | 4             | 3                     |

All data is lowercased except Google dataset. The model predictions stored in the dataset can be reproduced by following script.
```shell script
python analogy_test.py
```
When the model suffers out-of-vocabulary error, we use PMI prediction, `pmi_pred` in each entry, to ensure the baseline can
be compared with other methods to cover all the data points.   

Please read [our paper](https://arxiv.org/abs/2105.04949) for more information about the dataset and cite it if you use the dataset:
```
@inproceedings{ushio-etal-2021-bert-is,
    title ={{BERT} is to {NLP} what {A}lex{N}et is to {CV}: {C}an {P}re-{T}rained {L}anguage {M}odels {I}dentify {A}nalogies?},
    author={Ushio, Asahi and
            Espinosa-Anke, Luis and 
            Schockaert, Steven and
            Camacho-Collados, Jose},
    booktitle={Proceedings of the {ACL}-{IJCNLP} 2021 Main Conference},
    year={2021},
    publisher={Association for Computational Linguistics}
}
```

## Lexical Relation Dataset
Five different datasets for lexical relation classification used in [SphereRE](https://www.aclweb.org/anthology/P19-1169/).
This contains `BLESS`, `CogALexV`, `EVALution`, `K&H+N`, `ROOT09` and each of them has `test.tsv` and `train.tsv`.
Each tsv file consists of lines which describe the relation type given word A and B. 
```
A   B   relation_type
```
For more detailed discussion, please take a look the [SphereRE](https://www.aclweb.org/anthology/P19-1169/) paper.


To get word embedding baseline, 
```shell script
python lexical_relation.py
```
When the model suffers out-of-vocabulary error in evaluation, we use the most frequent label in training data, to ensure the baseline can
be compared with other methods to cover all the data points.   
 

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
Please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information about RELATIVE embedding.

