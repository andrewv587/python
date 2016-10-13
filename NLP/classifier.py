#!/usr/bin/python
#Filename:classifier.py
#Function:
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-10-12

import nltk
from nltk.corpus import names

import random
names1=[(name,'male') for name in names.words('male.txt')]
names2=[(name,'female') for name in names.words('female.txt')]
names=names1+names2
random.shuffle(names)

def gender_feature(name):
    return {'last_letter':name[-1]}

featuresets=[(gender_feature(n),g) for (n,g) in names]
train_set,test_set=featuresets[500:],featuresets[:500]
classifier=nltk.NaiveBayesClassifier.train(train_set)
#print classifier.classify(gender_feature('Trinity'))
print nltk.classify.accuracy(classifier,test_set)
print classifier.show_most_informative_features(10)

from nltk.corpus import brown
suffix_fdist=nltk.FreqDist()
for word in brown.words():
    word=word.lower()
    suffix_fdist[word[-1:]]+=1
    suffix_fdist[word[-2:]]+=1
    suffix_fdist[word[-3:]]+=1
common_sufixes = suffix_fdist.keys()[:100]
print common_sufixes
    
