import os
import logging
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
            full_data[os.path.basename(i)][os.path.basename(t).replace('.tsv', '')] = {'x': x, 'y': y}
        full_data[os.path.basename(i)]['label'] = label
    return full_data


def diff(a, b, model, add_feature_set='concat', relative_model=None, both_direction: bool = False):

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

    if relative_model is not None:
        try:
            vec_r = relative_model['__'.join([a, b]).lower().replace(' ', '_')]
        except KeyError:
            vec_r = np.zeros(relative_model.vector_size)
        feature.append(vec_r)
        if both_direction:
            try:
                vec_r = relative_model['__'.join([b, a]).lower().replace(' ', '_')]
            except KeyError:
                vec_r = np.zeros(relative_model.vector_size)
            feature.append(vec_r)
    return np.concatenate(feature)


def evaluate(embedding_model: str = None, feature_set='concat', add_relative: bool = False,
             both_direction: bool = False):
    model = get_word_embedding_model(embedding_model)
    model_re = None
    if add_relative:
        model_re = get_word_embedding_model('relative_init.{}'.format(embedding_model))

    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(embedding_model, data_name))
        label_dict = v.pop('label')
        x = [diff(a, b, model, feature_set, model_re, both_direction) for (a, b) in v['train']['x']]

        # initialize zero vector for OOV
        dim = len([_x for _x in x if _x is not None][0])
        x = [_x if _x is not None else np.zeros(dim) for _x in x]

        # train
        clf = MLPClassifier().fit(x, v['train']['y'])

        # test
        report_tmp = {'model': embedding_model, 'feature_set': feature_set, 'add_relative': add_relative,
                      'both_direction': both_direction, 'label_size': len(label_dict), 'data': data_name}
        for prefix in ['test', 'val']:
            if prefix not in v:
                continue
            logging.info('\t run {}'.format(prefix))
            x = [diff(a, b, model, feature_set, model_re, both_direction) for (a, b) in v[prefix]['x']]
            oov = sum([_x is None for _x in x])
            x = [_x if _x is not None else np.zeros(dim) for _x in x]
            y_pred = clf.predict(x)
            f_mac = f1_score(v[prefix]['y'], y_pred, average='macro')
            f_mic = f1_score(v[prefix]['y'], y_pred, average='micro')
            accuracy = sum([a == b for a, b in zip(v[prefix]['y'], y_pred.tolist())])/len(y_pred)
            report_tmp.update(
                {'accuracy/{}'.format(prefix): accuracy,
                 'f1_macro/{}'.format(prefix): f_mac,
                 'f1_micro/{}'.format(prefix): f_mic,
                 'oov/{}'.format(prefix): oov,
                 'data_size/{}'.format(prefix): len(y_pred)}
            )
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    del model_re
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
            # for if_relative in [True, False]:
            for if_relative in [False]:
                if not if_relative:
                    full_result += evaluate(m, feature_set=_feature, add_relative=if_relative)
                else:
                    for if_both in [True, False]:
                        full_result += evaluate(m, feature_set=_feature, add_relative=if_relative, both_direction=if_both)
        pd.DataFrame(full_result).to_csv(export)


