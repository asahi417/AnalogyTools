import os
import logging
from glob import glob
import tqdm
from itertools import product
from multiprocessing import Pool

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from util import get_word_embedding_model, wget
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
pbar = tqdm.tqdm()


def get_lexical_relation_data():
    """ get dataset """
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
            for _y in y:
                if _y not in label:
                    label[_y] = len(label)
            y = [label[_y] for _y in y]
            full_data[os.path.basename(i)][os.path.basename(t).replace('.tsv', '')] = {'x': x, 'y': y}
        full_data[os.path.basename(i)]['label'] = label
    return full_data


def diff(a, b, model, add_feature='concat', pair_models=None, bi_direction: bool = True):
    """ get feature for each pair """

    try:
        vec_a = model[a]
        vec_b = model[b]
    except KeyError:
        return None
    if 'concat' in add_feature:
        feature = [vec_a, vec_b]
    else:
        feature = []
    if 'diff' in add_feature:
        feature.append(vec_a - vec_b)
    if 'dot' in add_feature:
        feature.append(vec_a * vec_b)

    for pair_model in pair_models:
        if pair_model is not None:
            try:
                vec_r = pair_model['__'.join([a, b]).lower().replace(' ', '_')]
            except KeyError:
                vec_r = np.zeros(pair_model.vector_size)
            feature.append(vec_r)
            if bi_direction:
                try:
                    vec_r = pair_model['__'.join([b, a]).lower().replace(' ', '_')]
                except KeyError:
                    vec_r = np.zeros(pair_model.vector_size)
                feature.append(vec_r)
    return np.concatenate(feature)


def run_test(clf, x, y):
    """ run evaluation on valid or test set """
    y_pred = clf.predict(x)
    f_mac = f1_score(y, y_pred, average='macro')
    f_mic = f1_score(y, y_pred, average='micro')
    accuracy = sum([a == b for a, b in zip(y, y_pred.tolist())]) / len(y_pred)
    return accuracy, f_mac, f_mic


class Evaluate:

    def __init__(self, dataset, shared_config, default_config: bool = False):
        self.dataset = dataset
        if default_config:
            self.configs = [{'random_state': 0}]
        else:
            learning_rate_init = [0.001, 0.0001, 0.00001]
            # max_iter = [25, 50, 75]
            hidden_layer_sizes = [100, 150, 200]
            # learning_rate_init = [0.001, 0.0001]
            # max_iter = [25]
            # hidden_layer_sizes = [100]
            self.configs = [{
                'random_state': 0, 'learning_rate_init': i[0],
                'hidden_layer_sizes': i[2]} for i in
                            list(product(learning_rate_init, hidden_layer_sizes))]
        self.shared_config = shared_config

    @property
    def config_indices(self):
        return list(range(len(self.configs)))

    def __call__(self, config_id):
        pbar.update(1)
        config = self.configs[config_id]
        # train
        x, y = self.dataset['train']
        clf = MLPClassifier(**config).fit(x, y)
        # test
        x, y = self.dataset['test']
        t_accuracy, t_f_mac, t_f_mic = run_test(clf, x, y)
        report = self.shared_config.copy()
        report.update(
            {'metric/test/accuracy': t_accuracy,
             'metric/test/f1_macro': t_f_mac,
             'metric/test/f1_micro': t_f_mic,
             'classifier_config': clf.get_params()})
        if 'val' in self.dataset:
            x, y = self.dataset['val']
            v_accuracy, v_f_mac, v_f_mic = run_test(clf, x, y)
            report.update(
                {'metric/val/accuracy': v_accuracy,
                 'metric/val/f1_macro': v_f_mac,
                 'metric/val/f1_micro': v_f_mic})
        return report


def evaluate(embedding_model: str = None, feature='concat', add_relative: bool = False, add_pair2vec: bool = False):
    model = get_word_embedding_model(embedding_model)
    model_pair = []
    if add_relative:
        model_pair.append(get_word_embedding_model('relative_init.{}'.format(embedding_model)))
    if add_pair2vec:
        model_pair.append(get_word_embedding_model('pair2vec'))

    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(embedding_model, data_name))
        label_dict = v.pop('label')
        # preprocess data
        oov = {}
        dataset = {}
        for _k, _v in v.items():
            x = [diff(a, b, model, feature, model_pair) for (a, b) in _v['x']]
            dim = len([_x for _x in x if _x is not None][0])
            # initialize zero vector for OOV
            dataset[_k] = [
                [_x if _x is not None else np.zeros(dim) for _x in x],
                _v['y']]
            oov[_k] = sum([_x is None for _x in x])
        shared_config = {
            'model': embedding_model, 'feature': feature, 'add_relative': add_relative,
            'add_pair2vec': add_pair2vec, 'label_size': len(label_dict), 'data': data_name,
            'oov': oov
        }

        # grid serach
        if 'val' not in dataset:
            evaluator = Evaluate(dataset, shared_config, default_config=True)
            tmp_report = evaluator(0)
        else:
            pool = Pool()
            evaluator = Evaluate(dataset, shared_config)
            tmp_report = pool.map(evaluator, evaluator.config_indices)
            pool.close()
        tmp_report = [tmp_report] if type(tmp_report) is not list else tmp_report
        report += tmp_report
        # print(report)
        # print(pd.DataFrame(report))
        # input()
    del model
    del model_pair
    return report


if __name__ == '__main__':
    # model_name = os.getenv('MODEL', 'w2v')
    # print(model_name)
    # target_word_embedding = [model_name]
    target_word_embedding = ['w2v', 'fasttext', 'glove']
    done_list = []
    full_result = []
    export = 'results/lexical_relation_all.csv'
    if os.path.exists(export):
        df = pd.read_csv(export, index_col=0)
        done_list = df[['model', 'feature']].values.tolist()
        full_result = [i.to_dict() for _, i in df.iterrows()]
    logging.info("RUN WORD-EMBEDDING BASELINE")
    pattern = ['diff', 'concat', ('diff', 'dot'), ('concat', 'dot')]
    for m in target_word_embedding:
        for _feature in pattern:
            if [m, str(_feature)] in done_list:
                continue
            full_result += evaluate(m, feature=_feature)
            if _feature in [('diff', 'dot'), ('concat', 'dot')]:
                full_result += evaluate(m, feature=_feature, add_relative=True)
                full_result += evaluate(m, feature=_feature, add_pair2vec=True)
            pd.DataFrame(full_result).to_csv(export)
    # aggregate result
    # export = 'results/lexical_relation.{}.csv'.format(model_name)
    export = 'results/lexical_relation.csv'
    out = []
    df = pd.DataFrame(full_result)
    for _m in df.model.unique():
        for _f in df.feature.unique():
            for _d in df.data.unique():
                for _r in df.add_relative.unique():
                    for _p in df.add_pair2vec.unique():
                        df_tmp = df[df.model == _m][df.feature == _f][df.data == _d][df.add_relative
                                                                                     == _r][df.add_pair2vec == _p]
                        df_tmp = df_tmp.sort_values(by=['metric/val/f1_macro'], ascending=False)
                        out.append(df_tmp.head(1))
    pd.concat(out).to_csv(export)

