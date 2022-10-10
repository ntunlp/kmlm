import json
import gzip
import itertools

import numpy as np
import pymongo
from pymongo import MongoClient

MONGO_PATH='mongodb://127.0.0.1:27017/'
LANG='zh-hans'
LANG='vi'
DUMP_FILE='PATH/TO/kg_clean.cycle/' + LANG + '.txt'
ENTITY_PATH='PATH/TO/wiki5m/entities'
RELATION_PATH='PATH/TO/wiki5m/relation'
MAX_LINES = 50000000
CACHE = {}

def get_root_cycles3(root, edges):
    result = set()
    edge_dict = {}
    for e in edges:
        e1 = e[0]
        e2 = e[-1]
        if e1 == e2:
            continue
        for i in [e1, e2]:
            if i not in edge_dict:
                edge_dict[i] = {}
        edge_dict[e1][e2] = True
        edge_dict[e2][e1] = True
    if root not in edge_dict:
        return []
    immediate_conn = edge_dict[root]
    for i in immediate_conn:
        for j in edge_dict[i]:
            if j in immediate_conn:
                result.add(tuple(sorted([root, i, j])))
    return list(result)


def load_id(id_path):
    result = {}
    with open(id_path, 'r') as inf:
        for line in inf:
            result[line.strip()] = 1
    return result


def connectdb():
    client = MongoClient(MONGO_PATH)
    return client


def get_aliases(item, language, first_half=True):
    a = item['aliases'][language]
    length = len(a)
    if first_half and length > 2:
        length = int(length/2) + 1
    n = np.random.randint(length)
    return a[n]['value']


def get_lines(db, token, aliase=True, relation_id=None):
    result = []
    items = db['item'].find({'id': token})
    for item1 in items:
        if 'claims' not in item1:
            continue
        for pid in item1['claims']:
            if (relation_id is not None) and (not relation_id.get(pid)):
                continue
            prop = db['property'].find_one({'id': pid})
            for record in item1['claims'][pid]:
                try:
                    item2_id = record['mainsnak']['datavalue']['value']['id']
                    item2 = db['item'].find_one({'id': item2_id})
                    l1, l2, l3 = np.random.choice(LANGS, 3)
                    line = [item1['labels'][l1]['value'],
                            prop['labels'][l2]['value'],
                            item2['labels'][l3]['value'],
                           ]
                    result.append(' | '.join(line))
                    if aliase:
                        line = [get_aliases(item1, l1),
                                get_aliases(prop, l2),
                                get_aliases(item2, l3),
                               ]
                        result.append(' | '.join(line))
                except:
                    continue
    return result


def get_immediate_edges(db, item1_id, relation_id=None):
    result = []
    id2value = {}
    item1 = db['item'].find_one({'id': item1_id})
    id2value[item1_id] = item1['labels'][LANG]['value']
    for pid in item1['claims']:
        if (relation_id is not None) and (not relation_id.get(pid)):
            continue
        prop = db['property'].find_one({'id': pid})
        id2value[pid] = prop['labels'][LANG]['value']
        for record in item1['claims'][pid]:
            try:
                item2_id = record['mainsnak']['datavalue']['value']['id']
                item2 = db['item'].find_one({'id': item2_id})
                id2value[item2_id] = item2['labels'][LANG]['value']
                result.append([item1_id, pid, item2_id])
            except:
                continue
    return result, id2value


def get_entity2(edges):
    result = set()
    for e in edges:
        result.add(e[-1])
    return result

def get_triplet_dict(edges):
    result = {}
    for e in edges:
        result['_'.join(sorted([e[0], e[-1]]))] = e
    return result

def get_edges_in_cycle(line):
    result = []
    for i in range(len(line) - 1):
        result.append([line[i], line[i+1]])
    result.append([line[0], line[-1]])
    return result

def run():
    client = connectdb()
    db = client['wikidata']
    i = 0
    entity_id = load_id(ENTITY_PATH)
    relation_id = load_id(RELATION_PATH)
    with open(DUMP_FILE, 'w') as outf:
        for item_id in entity_id:
            try:
                edges, id2value = get_immediate_edges(db, item_id, relation_id=relation_id)
                children = get_entity2(edges)
                for item_id in children:
                    c_edges, c_id2value = get_immediate_edges(db, item_id, relation_id=relation_id)
                    edges += c_edges
                    id2value.update(c_id2value)
            except:
                continue
            triplet_dict = get_triplet_dict(edges)
            for line in get_root_cycles3(item_id, edges):
                if line in CACHE:
                    continue
                else:
                    CACHE[line] = True
                sentences = []
                for comb in get_edges_in_cycle(line):
                    comb = '_'.join(sorted(comb))
                    if comb not in triplet_dict:
                        continue
                    t = triplet_dict[comb]
                    sentences.append('| '.join([id2value[t[0]], id2value[t[1]], id2value[t[2]]]).replace('.', '') + '. ')
                outf.write(' '.join(sentences) + '\n')
                i += 1
                if i % 10000 == 0:
                    print('lines written: ', i)
            if i > MAX_LINES:
                break
    client.close()

run()


