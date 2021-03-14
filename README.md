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
 

- ***script to train RELATIVE model***: [`get_relative_embedding.py`](./get_relative_embedding.py)




## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.