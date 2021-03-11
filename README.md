# Analogy Dataset
Dataset for analogical knowledge probing:
- SAT-style analogy test:
    - [SAT](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/sat.zip)
    - [U2](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u2.zip)
    - [U4](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/u4.zip)
    - [Google](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/google.zip)
    - [BATS](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/bats.zip)
- Word pair list:
    - [wiki-common-word-pairs](https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0/relative_vocab.tar.gz)

## Scripts
- [`get_pair_vocab.py`](./get_pair_vocab.py): Retrieve wiki-common-word-pairs from [relative embedding model](https://github.com/pedrada88/relative), which
is a word-pair embedding model trained on selected word pair from Wikipedia dump.

- [`get_relative_embedding.py`](./get_relative_embedding.py): Compute [relative embedding](http://josecamachocollados.com/papers/relative_ijcai2019.pdf)
over Wikipedia with FastText.
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
Please refer [the official implementation](https://github.com/pedrada88/relative) and [the paper](http://josecamachocollados.com/papers/relative_ijcai2019.pdf)
for further information related to relative embedding.