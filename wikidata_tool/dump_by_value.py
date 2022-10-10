import json
import gzip

import numpy as np
import pymongo
from pymongo import MongoClient

MONGO_PATH='mongodb://127.0.0.1:27017/'
DUMP_FILE='./file.txt'
LANGS=['en', 'zh-hans']

def connectdb():
    client = MongoClient(MONGO_PATH)
    return client


def get_lines(db, token):
    result = []
    items = db['item'].find({'labels.en.value': token})
    for item1 in items:
        if 'claims' not in item1:
            continue
        for pid in item1['claims']:
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
                    result.append('|'.join(line))
                except:
                    continue
    return result


def run():
    client = connectdb()
    db = client['wikidata']
    i = 0
    with open(DUMP_FILE, 'w') as outf:
        for token in ['happiness', 'apple', 'Singapore', 'Google']:
            for line in get_lines(db, token):
                outf.write(line + '\n')
                i += 1
                if i % 10000 == 0:
                    print('lines written: ', i)
    client.close()

run()


