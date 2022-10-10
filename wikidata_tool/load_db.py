import json
import gzip

import pymongo
from pymongo import MongoClient

MONGO_PATH='mongodb://localhost:27017/'
RAW_FILE_PATH='PATH/TO/wikidata/wikidata-20210301-all.json.gz'

def connectdb():
    client = MongoClient(MONGO_PATH)
    return client

def get_record():
    with gzip.open(RAW_FILE_PATH,'r') as inf:
        for line in inf:
            line = line.decode("utf-8").strip()
            if line[-1] == ',':
                line = line[:-1]
            if len(line) < 5:
                continue
            try:
                line = json.loads(line)
                line['sitelinks'] = 0
                line['lastrevid'] = 0
                if 'en' in line['labels']:
                    line['en_label'] = line['labels']['en']['value']
            except:
                print('error processing:\n', line)
                continue
            yield line

def insert_data(db, line):
    if line['type'] == 'property':
        db['property'].insert_one(line)
    elif line['type'] == 'item':
        db['item'].insert_one(line)


def run():
    client = connectdb()
    db = client['wikidata']
    i = 0
    for line in get_record():
        try:
            insert_data(db, line)
            i += 1
            if i % 10000 == 0:
                print('lines loaded: ', i)
        except:
            print('fail to load ', line)
            continue
    client.close()

run()


