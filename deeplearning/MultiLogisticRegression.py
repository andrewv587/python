#!/usr/bin/python
#Filename:MultiLogisticRegression
#Function:make a MultiLogisticRegression 
# class model to fulfill MultiLogisticRegression
# function
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-02

import numpy as np
import theano
from theano import shared
from theano import function
import theano.tensor as T

class MultiLogisticRegression:
    #data(n,m) n represents number of samples
    #data_m represents datapoints length(in)
    #pred_m represents class number(out)
    #W B model parameters
    def __init__(self,x,data_m,pred_m):
        self.W = shared(np.zeros((data_m,pred_m)
            ,dtype=theano.config.floatX))
        self.b = shared(np.zeros(pred_m
            ,dtype=theano.config.floatX))
        self.yij= T.nnet.softmax(T.dot(x,self.W)+self.b)
        self.pred = T.argmax(self.yij,axis=1)
        self.params=[self.W,self.b]

    def loss(self,y):
        return -T.mean(T.log(self.yij[T.arange(y.shape[0]),y]))

    def grad(self,y):
        error=self.loss(y)
        gw,gb=T.grad(error,[self.W,self.b])
        return gw,gb

    #update W and B
    def update(self,y,alpha):
        gw,gb=self.grad(y)
        updates=[(self.W,self.W-alpha*gw),(self.b,self.b-alpha*gb)]
        return updates

    def error(self,y):
        return T.mean(T.neq(self.pred,y))
    
#itr:iteration num
#alpha: step length
def run_test(datafile='data/mnist.pkl.gz',itr=400,alpha=0.1):
    import cPickle,gzip
    with gzip.open(datafile,'rb') as f:
        train_set,valid_set,test_set=cPickle.load(f)
    trainx,trainy=train_set
    validx,validy=valid_set
    testx,testy=test_set
    x = T.fmatrix('x')
    y = T.lvector('y')
    classifer =MultiLogisticRegression(x,28*28,10)
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
        
