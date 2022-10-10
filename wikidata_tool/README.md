# tools for Wikidata preprocessing

The tools can be used to first load data to mongodb, and then run queries to generate data in the desired format.

## download data
```
wget https://dumps.wikimedia.org/wikidatawiki/entities/20210301/wikidata-20210301-all.json.gz
```

## create mongodb
```
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel62-4.2.13.tgz
mongo/bin/mongod -f mongo_config.ini
mongo/bin/mongo 127.0.0.1:27017

# build index for query speed up
> db.item.createIndex({'labels.en.value': 1})
> db.item.createIndex({'id': 1})
> db.property.createIndex({'id': 1})
```
## load to db
```
python load_db.py
```

- entity keys
```
['type', 'id', 'labels', 'descriptions', 'aliases', 'claims', 'sitelinks', 'lastrevid', 'en_label']
```

- property keys
```
['type', 'datatype', 'id', 'labels', 'descriptions', 'aliases', 'claims', 'lastrevid', 'en_label']
```

## query example
```
mongo 11.139.194.169:27017
> use wikidata
> show collections
> db.item.find({'labels.en.value': 'happiness'}, {'id': 1, 'labels.en.value': 1})
{ "_id" : ObjectId("605c112a1bc6e1ed4bb5f042"), "id" : "Q8", "labels" : { "en" : { "value" : "happiness" } } }
> db.property.find({}, {'id': 1, 'en_label':1, 'labels.zh': 1})
> db.item.find({'labels.en.value': 'happiness'}, {'id': 1, 'labels.en.value': 1, 'claims.P31': 1}).pretty()
```

## other
pretraining data for xlmr http://data.statmt.org/cc-100/


## Sample filtered relations and entities from kepler
```
head PATH/TO/wiki5m/entities
Q1
Q100
Q1000
Q10000
Q100000
Q1000000
Q1000001
Q1000003
Q1000004
Q1000005

head PATH/TO/wiki5m/relation
P1000
P1001
P1002
P101
P1011
P1013
P1018
P102
P1026
P1027
```
