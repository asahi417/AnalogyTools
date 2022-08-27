import os
import json
from itertools import permutations
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import pandas as pd

from util import get_word_embedding_model, wget


def get_data():
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    url = "https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/relation_mapping_data.jsonl"
    if not os.path.exists(f'{cache_dir}/relation_mapping_data.jsonl'):
        wget(url, cache_dir)
    with open(f'{cache_dir}/relation_mapping_data.jsonl') as f:
        return [json.loads(i) for i in f.read().split('\n') if len(i) > 0]


def embedding_model(model_name):
    if model_name in ['fasttext', 'fasttext_cc']:
        model = get_word_embedding_model(model_name)
        def get_embedding(a, b): return (model[a] - model[b]).tolist()
    else:
        raise ValueError(f'unknown model {model_name}')
    return get_embedding


def cache_embedding():
    # get data
    data = get_data()

    os.makedirs('embeddings', exist_ok=True)
    for m in ['fasttext', 'fasttext_cc']:
        embeder = embedding_model(m)
        for data_id, _data in enumerate(data):
            print(f'[{m}]: {data_id}/{len(data)}')
            cache_file = f'embeddings/{m}.vector.{data_id}.json'
            embedding_dict = {}
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    embedding_dict = json.load(f)
            for _type in ['source', 'target']:
                for x, y in permutations(_data[_type], 2):
                    _id = f'{x}__{y}'
                    if _id not in embedding_dict:
                        vector = embeder(x, y)
                        embedding_dict[_id] = vector
                        with open(cache_file, 'w') as f_writer:
                            json.dump(embedding_dict, f_writer)


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-4)


def mean(_list):
    return sum(_list)/len(_list)


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

    accuracy_full = {}
    for m in ['relbert', 'fasttext_cc']:
        accuracy = []
        sims_full = []
        perms_full = []
        for data_id, _data in enumerate(data):
            print(f'[{m}]: {data_id}/{len(data)}')
            with open(f'embeddings/{m}.vector.{data_id}.json') as f:
                embedding_dict = json.load(f)
            sim_file = f'embeddings/{m}.sim.{data_id}.json'
            sim = {}
            sim_flag = {}
            if os.path.exists(sim_file):
                with open(sim_file) as f:
                    sim = json.load(f)

            source = _data['source']
            target = _data['target']
            perms = []
            for n, tmp_target in tqdm(list(enumerate(permutations(target, len(target))))):
                list_sim = []
                for id_x, id_y in permutations(range(len(target)), 2):
                    _id = f'{source[id_x]}__{source[id_y]} || {tmp_target[id_x]}__{tmp_target[id_y]}'
                    if _id not in sim:
                        sim[_id] = cosine_similarity(
                            embedding_dict[f'{source[id_x]}__{source[id_y]}'],
                            embedding_dict[f'{tmp_target[id_x]}__{tmp_target[id_y]}']
                        )
                        with open(sim_file, 'w') as f_writer:
                            json.dump(sim, f_writer)
                    if target[id_x] == tmp_target[id_x] and target[id_y] == tmp_target[id_y]:
                        sim_flag[_id] = True
                    else:
                        sim_flag[_id] = False
                    list_sim.append(sim[_id])
                perms.append({'target': tmp_target, 'similarity_mean': mean(list_sim)})
            sims_full.extend([{'pair': k, 'sim': v, 'is_analogy': sim_flag[k], 'data_id': data_id}
                              for k, v in sim.items()])
            pred = sorted(perms, key=lambda _x: _x['similarity_mean'], reverse=True)
            accuracy.extend([t == p for t, p in zip(target, pred[0]['target'])])
            tmp = [i for i in perms if list(i['target']) == target]
            assert len(tmp) == 1, perms
            perms_full.append({
                'source': source,
                'true': target, 'pred': pred[0]['target'], 'accuracy': list(pred[0]['target']) == target,
                'similarity': pred[0]['similarity_mean'], 'similarity_true': tmp[0]['similarity_mean']
            })
        pd.DataFrame(sims_full).to_csv(f'./output/stats.sim.{m}.csv')
        pd.DataFrame(perms_full).to_csv(f'./output/stats.breakdown.{m}.csv')

        accuracy_full[m] = mean(accuracy)

    print(json.dumps(accuracy_full, indent=4))
    with open('output/result.json', 'w') as f:
        json.dump(accuracy_full, f)
