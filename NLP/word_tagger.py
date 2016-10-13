#!/usr/bin/python
#Filename:test.py
#Function:
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-10-10

import nltk
from nltk.corpus import brown

brown_tagged_sents=brown.tagged_sents(categories='news')
brown_tagged_words=brown.tagged_words(categories='news')
brown_sents=brown.sents(categories='news')

default_tagger=nltk.DefaultTagger('NN')
unigram_tagger=nltk.UnigramTagger(brown_tagged_sents[:4160],backoff=default_tagger)
#print unigram_tagger.tag(brown_sents[2007])
#print unigram_tagger.evaluate(brown_tagged_sents[4160:])
bigram_tagger=nltk.BigramTagger(brown_tagged_sents[:4160],backoff=unigram_tagger)
#print bigram_tagger.tag(brown_sents[4203])
#print bigram_tagger.evaluate(brown_tagged_sents[4160:])



#tags=[tag for (word,tag) in brown_tagged_words]
#print nltk.FreqDist(tags).max()
#
#raw='I do not like green eggs and ham,I do not like them Sam I am!'
#tokens=nltk.word_tokenize(raw)
#default_tagger=nltk.DefaultTagger('NN')
#print default_tagger.tag(tokens)
#
##print default_tagger.evaluate(brown_tagged_sents)
#
#fd=nltk.FreqDist(brown.words(categories='news'))
#cfd=nltk.ConditionalFreqDist(brown_tagged_words)
#print cfd
#most_freq_words=fd.keys()[:100]
#print most_freq_words
#likely_tags=dict((word,cfd[word].max()) for word in most_freq_words)
#baseline_tagger=nltk.UnigramTagger(model=likely_tags)
##print baseline_tagger.evaluate(brown_tagged_sents)
#sent=brown_sents[3]
#print baseline_tagger.tag(sent)
