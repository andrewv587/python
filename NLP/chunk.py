#!/usr/bin/python
#Filename:chunk.py
#Function:
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-10-13

import nltk,re,pprint
sentence=[("the","DT"),("little","JJ"),("yellow","JJ"),("dog","NN"), \
    ("barked","VBD"),("at","IN"),("the","DT"),("cat","NN")]

#grammar = "NP:{<DT>?<JJ>*<NN>}"
#grammar=r"""
#    NP:{<DT|PP\$>?<JJ>*<NN>}
#    {<NNP>+}
#"""
grammar=r"""
    NP:{<.*>+}
    }<VBD|IN>+{
"""
cp=nltk.RegexpParser(grammar)
result=cp.parse(sentence)
print result
#result.draw()

cp=nltk.RegexpParser('NOUNS: {<V.*><TO><V.*>}')
brown=nltk.corpus.brown
i=0
for sent in brown.tagged_sents():
    tree=cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label()=='NOUNS' :
            print subtree
            i+=1
    if i>10:
        break

from nltk.corpus import conll2000
grammar=r"NP:{<[CDJNP].*>+}"
cp=nltk.RegexpParser(grammar)
test_sents=conll2000.chunked_sents('test.txt',chunk_types=['NP'])
print cp.evaluate(test_sents)

class UnigramChunker(nltk.ChunkParserI):

    def __init__(self,train_sents):
        train_data=[[(t,c) for w,t,c in nltk.tree2conlltags(sent)] \
                for sent in train_sents]
        self.tagger=nltk.BigramTagger(train_data)

    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags=self.tagger.tag(pos_tags)
        chunktags=[chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags=[(word,pos,chunktag) for ((word,pos),chunktag) \
                in zip(sentence,chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

train_sents=conll2000.chunked_sents('train.txt',chunk_types=['NP'])
unigram_chunker=UnigramChunker(train_sents)
print unigram_chunker.evaluate(test_sents)



#class ConsecutiveNPChunkTagger(nltk.TaggerI):
#    def __init__(self,train_sents):
#        train_set=[]
#        def npchunk_features(sentence,i,history):
#            word,pos = sentence[i]
#            return {"pos":pos}
#        print "aa"
#        for tagged_sent in train_sents:
#            untagged_sent=nltk.tag.untag(tagged_sent)
#            history=[]
#            for i,(word,tag) in enumerate(tagged_sent):
#                featureset=npchunk_features(untagged_sent,i,history)
#                train_set.append((featureset,tag))
#                history.append(tag)
#        self.classfifier=nltk.MaxentClassfier.train(train_set,algorithm='megam',trace=0)
#    def tag(self,sentence):
#        history=[]
#        for i,word in enumerate(sentence):
#            featureset=npchunk_features(sentence,i,history)
#            tag=self.classfifier.classfify(featureset)
#            history.append(tag)
#        return zip(sentence,history)
#
#class ConsecutiveNPChunker(nltk.ChunkParserI):
#
#    def __init__(self,train_sents):
#        print "def"
#        tagged_sents=[[((w,t),c) for w,t,c in nltk.tree2conlltags(sent)] \
#                for sent in train_sents]
#        self.tagger=ConsecutiveNPChunker(tagged_sents)
#        print "abc"
#
#    def parse(self,sentence):
#        tagged_sents=self.tagger.tag(sentence)
#        conlltags=[(w,t,c) for ((w,t),c) in tagged_sents]
#        return nltk.chunk.conlltags2tree(conlltags)
#
#
#chunker=ConsecutiveNPChunker(train_sents)
#print chunker.evaluate(test_sents)


