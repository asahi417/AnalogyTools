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
We train RELATIVE model on Fasttext model trainedd  

- ***script to reproduce the data***: [`get_relative_embedding.py`](./get_relative_embedding.py)
Compute [relative embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf)
over Wikipedia with FastText for the word pairs of analogy test dataset and wiki-common-word-pairs.
```shell script
usage: get_relative_embedding.py [-h] [-o OUTPUT] [-w WINDOW_SIZE] [--minimum-frequency-context MINIMUM_FREQUENCY_CONTEXT] [--minimum-frequency MINIMUM_FREQUENCY]

simplified RELATIVE embedding training

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path to store relation vectors
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        Co-occurring window size
  --minimum-frequency-context MINIMUM_FREQUENCY_CONTEXT
                        Minimum frequency of words between word pair: increasing the number can speed up thecalculations and reduce memory but we would recommend keeping this number low
  --minimum-frequency MINIMUM_FREQUENCY
                        Minimum frequency of words
```


## Acknowledgement
About RELATIVE embedding work, please refer [the official implementation](https://github.com/pedrada88/relative) and
[the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf) for further information.