#!/usr/bin/python
#Filename:conn_mongodb.py
#Function:this function is to connect mongodb
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-07

#!/usr/bin/python

import pymongo
import random

client = pymongo.MongoClient("docker-ubuntu",27017)
db = client.test
db.user.drop()
db.user.save({'id':1,'name':'kaka','sex':'male'})
for id in range(2,10):
    name = random.choice(['steve','koby','owen','tody','rony'])
    sex = random.choice(['male','female'])
    db.user.insert({'id':id,'name':name,'sex':sex}) 
content = db.user.find()
for i in content:
     print i
