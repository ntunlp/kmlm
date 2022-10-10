import json
import gzip

import numpy as np
import pymongo
from pymongo import MongoClient

MONGO_PATH='mongodb://127.0.0.1:27017/'
LANGS=['en', 'ko']
DUMP_FILE='PATH/TO/kg_clean/' + '_'.join(LANGS) + '.txt'
ENTITY_PATH='PATH/TO/wiki5m/entities'
RELATION_PATH='PATH/TO/wiki5m/relation'

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
                    result.append('| '.join(line) + '. ')
                    if aliase:
                        line = [get_aliases(item1, l1),
                                get_aliases(prop, l2),
                                get_aliases(item2, l3),
                               ]
                        result.append('| '.join(line) + '. ')
                except:
                    continue
    return result


def run():
    client = connectdb()
    db = client['wikidata']
    i = 0
    entity_id = load_id(ENTITY_PATH)
    relation_id = load_id(RELATION_PATH)
    with open(DUMP_FILE, 'w') as outf:
        for token in entity_id:
            for line in get_lines(db, token, relation_id=relation_id):
                outf.write(line + '\n')
                i += 1
                if i % 10000 == 0:
                    print('lines written: ', i)
    client.close()

run()


