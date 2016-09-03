#!/usr/bin/python
#Filename:MultipleLayerPerception.py
#Function:fufill MLP function
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-03
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
from MultiLogisticRegression import MultiLogisticRegression
from MiddleLayer import MiddleLayer
class MultipleLayerPerception:
    def __init__(self,x,n_in,n_middle,n_out):
        self.mdl=MiddleLayer(x,n_in,n_middle)
        self.mlg=MultiLogisticRegression(self.mdl.y,n_middle,n_out)
        self.params=[self.mdl.W,self.mdl.b,self.mlg.W,self.mlg.b]
        self.pred=self.mlg.pred
    
    def loss(self,y):
        return self.mlg.loss(y)

    def update(self,y,alpha):
        misfit=self.loss(y)
        gradparams=[]
        for param in self.params:
           tmpParam=T.grad(misfit,param) 
           gradparams.append(tmpParam)
        updates=[]
        for param,gpram in zip(self.params,gradparams):
            updates.append((param,param-gpram*alpha))
        return updates

    def error(self,y): 
        return T.mean(T.neq(self.pred,y))

#itr:iteration num
#alpha: step length
def run_test(datafile='data/mnist.pkl.gz',itr=40,alpha=0.1):
    import cPickle,gzip
    with gzip.open(datafile,'rb') as f:
        train_set,valid_set,test_set=cPickle.load(f)
    trainx,trainy=train_set
    validx,validy=valid_set
    testx,testy=test_set
    x = T.fmatrix('x')
    y = T.lvector('y')
    classifer =MultipleLayerPerception(x,28*28,300,10)
    print testx
    updates=classifer.update(y,alpha)
    lg_model=function([x,y],classifer.loss(y),updates=updates)
    lg_error=function([x,y],classifer.error(y))
    lg_pred=function([x],classifer.pred)
    for i in range(itr):
        print i
        train_loss=lg_model(trainx,trainy)
    train_error=lg_error(validx,validy)
    train_pred=lg_pred(testx)
    print train_error
    print "predict and true class"
    for i in xrange(testy.shape[0]):
        print train_pred[i],testy[i]
    
    print "test error"
    test_error=(train_pred!=testy).mean()
    print test_error
        

if __name__ == '__main__':
    run_test()

