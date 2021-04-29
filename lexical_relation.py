import os
import logging
import argparse
from itertools import combinations, groupby
from glob import glob
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from util import get_word_embedding_model, wget
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_lexical_relation_data():
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.tar.gz'
    if not os.path.exists('{}/lexical_relation_dataset'.format(cache_dir)):
        wget(root_url_analogy, cache_dir)
    full_data = {}
    for i in glob('{}/lexical_relation_dataset/*'.format(cache_dir)):
        if not os.path.isdir(i):
            continue
        full_data[os.path.basename(i)] = {}
        label = {}
        for t in glob('{}/*tsv'.format(i)):
            with open(t) as f:
                data = [line.split('\t') for line in f.read().split('\n') if len(line) > 0]
            x = [d[:2] for d in data]
            y = [d[-1] for d in data]
            for _y in list(set(y)):
                if _y not in label:
                    label[_y] = len(label)
            y = [label[_y] for _y in y]
            full_data[os.path.basename(i)][os.path.basename(t).replace('.tsv', '')] = {'x': x, 'y':y}
        full_data[os.path.basename(i)]['label'] = label
    return full_data


def diff(a, b, model, add_feature_set='concat'):
    try:
        vec_a = model[a]
        vec_b = model[b]
    except KeyError:
        return None
    if 'concat' in add_feature_set:
        feature = [vec_a, vec_b]
    else:
        feature = []
    if 'diff' in add_feature_set:
        feature.append(vec_a - vec_b)
    if 'dot' in add_feature_set:
        feature.append(vec_a * vec_b)
    return np.concatenate(feature)


def evaluate(embedding_model: str = None, feature_set='concat'):

    model = get_word_embedding_model(embedding_model)
    model_name = embedding_model
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(model_name, data_name))
        label_dict = v.pop('label')
        freq_label = sorted([(n, len(list(i))) for n, i in groupby(sorted(v['train']['y']))],
                            key=lambda o: o[1], reverse=True)[0][0]

        x = [diff(a, b, model, feature_set) for (a, b) in v['train']['x']]
        y = [y for y, flag in zip(v['train']['y'], x) if flag is not None]
        x = [_x for _x in x if _x is not None]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, y)

        logging.info('\t run validation')
        x = [diff(a, b, model, feature_set) for (a, b) in v['test']['x']]
        y_pred = np.ones(len(x)) * freq_label
        x = [(n, _x) for n, _x in enumerate(x) if _x is not None]
        y_pred_ = clf.predict([_x for _, _x in x])
        for _n, (n, _) in enumerate(x):
            y_pred[n] = y_pred_[_n]
        oov = len(y_pred) - len(x)

        f_mac = f1_score(v['test']['y'], y_pred, average='macro')
        f_mic = f1_score(v['test']['y'], y_pred, average='micro')
        accuracy = sum([a == b for a, b in zip(v['test']['y'], y_pred.tolist())])/len(y_pred)

        report_tmp = {'model': model_name, 'accuracy': accuracy, 'f1_macro': f_mac, 'f1_micro': f_mic,
                      'feature_set': feature_set,
                      'label_size': len(label_dict), 'oov': oov,
                      'test_total': len(y_pred), 'data': data_name}
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    return report


if __name__ == '__main__':
    target_word_embedding = ['w2v', 'glove', 'fasttext']
    done_list = []
    full_result = []
    export = 'results/lexical_relation.csv'
    if os.path.exists(export):
        df = pd.read_csv(export, index_col=0)
        done_list = list(set(df['model'].values))
        full_result = [i.to_dict() for _, i in df.iterrows()]
    logging.info("RUN WORD-EMBEDDING BASELINE")
    pattern = list(combinations(['diff', 'concat', 'dot'], 2)) + [('diff', 'concat', 'dot')] + ['diff', 'concat', 'dot']
    for m in target_word_embedding:
        if m in done_list:
            continue
        for _feature in pattern:
            full_result += evaluate(m, feature_set=_feature)
        pd.DataFrame(full_result).to_csv(export)


