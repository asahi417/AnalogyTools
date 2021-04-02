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
Common word pair dataset is a dataset consisting of a pair of head and tail word ([link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl)).
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

Additionally, a common word list taken by the intersection of glove and word2vec pretrained model is released
([link](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word.pkl)).

- ***script to reproduce the data***: [`generate_word_pair_dataset.py`](generate_word_pair_dataset.py)

## Pretrained Relation Embedding Model
Following [RELATIVE embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) models that trained on 
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
`{}_truecase` means the wikidump is converted into truecase by third party truecaser.
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

| data   | model                                  | oov_test | accuracy_test       | oov_valid | accuracy_valid      | accuracy            | 
|--------|----------------------------------------|----------|---------------------|-----------|---------------------|---------------------| 
| bats   | fasttext                               | 0        | 0.7065036131183992  | 0         | 0.7336683417085427  | 0.7092092092092092  | 
| bats   | glove                                  | 4        | 0.6876042245692051  | 0         | 0.6984924623115578  | 0.6886886886886887  | 
| bats   | relative_init.fasttext.concat          | 399      | 0.5503057254030017  | 50        | 0.5728643216080402  | 0.5525525525525526  | 
| bats   | relative_init.fasttext.truecase.concat | 399      | 0.5352973874374652  | 50        | 0.5326633165829145  | 0.535035035035035   | 
| bats   | relative_init.glove.concat             | 407      | 0.546414674819344   | 50        | 0.5829145728643216  | 0.55005005005005    | 
| bats   | relative_init.glove.truecase.concat    | 411      | 0.5419677598665925  | 54        | 0.542713567839196   | 0.5420420420420421  | 
| bats   | relative_init.w2v.concat               | 521      | 0.4802668148971651  | 61        | 0.49748743718592964 | 0.481981981981982   | 
| bats   | relative_init.w2v.truecase.concat      | 449      | 0.5158421345191774  | 58        | 0.5577889447236181  | 0.52002002002002    | 
| bats   | w2v                                    | 122      | 0.6186770428015564  | 11        | 0.6030150753768844  | 0.6171171171171171  | 
| google | fasttext                               | 0        | 0.948               | 0         | 0.94                | 0.9472727272727273  | 
| google | glove                                  | 0        | 0.96                | 0         | 0.92                | 0.9563636363636364  | 
| google | relative_init.fasttext.concat          | 65       | 0.724               | 6         | 0.72                | 0.7236363636363636  | 
| google | relative_init.fasttext.truecase.concat | 65       | 0.758               | 6         | 0.7                 | 0.7527272727272727  | 
| google | relative_init.glove.concat             | 73       | 0.764               | 7         | 0.64                | 0.7527272727272727  | 
| google | relative_init.glove.truecase.concat    | 67       | 0.782               | 6         | 0.72                | 0.7763636363636364  | 
| google | relative_init.w2v.concat               | 160      | 0.58                | 15        | 0.5                 | 0.5727272727272728  | 
| google | relative_init.w2v.truecase.concat      | 67       | 0.77                | 6         | 0.74                | 0.7672727272727272  | 
| google | w2v                                    | 0        | 0.932               | 0         | 0.9                 | 0.9290909090909091  | 
| sat    | fasttext                               | 0        | 0.4629080118694362  | 0         | 0.5405405405405406  | 0.47058823529411764 | 
| sat    | glove                                  | 0        | 0.47774480712166173 | 0         | 0.5945945945945946  | 0.4893048128342246  | 
| sat    | relative_init.fasttext.concat          | 88       | 0.2997032640949555  | 4         | 0.43243243243243246 | 0.3128342245989305  | 
| sat    | relative_init.fasttext.truecase.concat | 88       | 0.2997032640949555  | 4         | 0.4864864864864865  | 0.31818181818181823 | 
| sat    | relative_init.glove.concat             | 88       | 0.32047477744807124 | 4         | 0.43243243243243246 | 0.3315508021390374  | 
| sat    | relative_init.glove.truecase.concat    | 91       | 0.32047477744807124 | 4         | 0.43243243243243246 | 0.3315508021390374  | 
| sat    | relative_init.w2v.concat               | 89       | 0.3353115727002967  | 5         | 0.43243243243243246 | 0.3449197860962567  | 
| sat    | relative_init.w2v.truecase.concat      | 94       | 0.314540059347181   | 4         | 0.3783783783783784  | 0.32085561497326204 | 
| sat    | w2v                                    | 3        | 0.41543026706231456 | 0         | 0.5135135135135135  | 0.42513368983957217 | 
| u2     | fasttext                               | 0        | 0.3815789473684211  | 0         | 0.2916666666666667  | 0.373015873015873   | 
| u2     | glove                                  | 1        | 0.4605263157894737  | 0         | 0.375               | 0.4523809523809524  | 
| u2     | relative_init.fasttext.concat          | 49       | 0.3333333333333333  | 5         | 0.4166666666666667  | 0.3412698412698413  | 
| u2     | relative_init.fasttext.truecase.concat | 49       | 0.32456140350877194 | 5         | 0.2916666666666667  | 0.32142857142857145 | 
| u2     | relative_init.glove.concat             | 50       | 0.42105263157894735 | 5         | 0.25                | 0.40476190476190477 | 
| u2     | relative_init.glove.truecase.concat    | 50       | 0.3684210526315789  | 6         | 0.2916666666666667  | 0.3611111111111111  | 
| u2     | relative_init.w2v.concat               | 50       | 0.40350877192982454 | 5         | 0.375               | 0.4007936507936508  | 
| u2     | relative_init.w2v.truecase.concat      | 51       | 0.3684210526315789  | 6         | 0.4583333333333333  | 0.376984126984127   | 
| u2     | w2v                                    | 1        | 0.3991228070175439  | 0         | 0.3333333333333333  | 0.39285714285714285 | 
| u4     | fasttext                               | 0        | 0.38425925925925924 | 0         | 0.3958333333333333  | 0.3854166666666667  | 
| u4     | glove                                  | 3        | 0.4027777777777778  | 0         | 0.4583333333333333  | 0.4083333333333333  | 
| u4     | relative_init.fasttext.concat          | 83       | 0.32175925925925924 | 8         | 0.4166666666666667  | 0.33125             | 
| u4     | relative_init.fasttext.truecase.concat | 83       | 0.33564814814814814 | 8         | 0.375               | 0.33958333333333335 | 
| u4     | relative_init.glove.concat             | 84       | 0.33101851851851855 | 8         | 0.4583333333333333  | 0.34375             | 
| u4     | relative_init.glove.truecase.concat    | 85       | 0.3333333333333333  | 8         | 0.4166666666666667  | 0.3416666666666667  | 
| u4     | relative_init.w2v.concat               | 86       | 0.3263888888888889  | 8         | 0.3333333333333333  | 0.32708333333333334 | 
| u4     | relative_init.w2v.truecase.concat      | 87       | 0.3449074074074074  | 8         | 0.375               | 0.34791666666666665 | 
| u4     | w2v                                    | 5        | 0.3912037037037037  | 0         | 0.3541666666666667  | 0.3875              | 

The prediction from each model is exported at [here](./predictions). 

- ***script to reproduce the result***: [`test_analogy.py`](analogy_test.py)

## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.
For the fasttext model, please refer [the facebook release](https://fasttext.cc/docs/en/english-vectors.html).
