#!/usr/bin/python
#Filename:LeNet.py
#Function:fulfill LeNet
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-03

import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
from MultiLogisticRegression import MultiLogisticRegression
from MiddleLayer import MiddleLayer
from CNN import CNN
class LeNet:
    def __init__(self,x):
        self.cnn1 = CNN(x,x.shape,(20,1,5,5))
        self.cnn2 = CNN(self.cnn1.output,self.cnn1.output.shape,(50,20,5,5))
        middle_input=self.cnn2.output.flatten(2)
        self.mdl = MiddleLayer(middle_input,50*4*4,500)
        self.mlg = MultiLogisticRegression(self.mdl.y,500,10)
        self.pred = self.mlg.pred
        self.params=self.cnn1.params+self.cnn2.params+self.mdl.params+self.mlg.params

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
def run_test(datafile='data/mnist.pkl.gz',itr=100,alpha=0.05):
    import cPickle,gzip
    with gzip.open(datafile,'rb') as f:
        train_set,valid_set,test_set=cPickle.load(f)
    trainx,trainy=train_set
    validx,validy=valid_set
    testx,testy=test_set
    trainx=trainx[:15000,:]
    trainy=trainy[:15000]
    print trainx.shape
    print trainy.shape
    print validx.shape
    x=T.fmatrix('x')
    y=T.lvector('y')
    x0= x.reshape((x.shape[0],1,28,28))

    leNet5 = LeNet(x0)
    updates=leNet5.update(y,alpha)
    leNet5_model=function([x,y],leNet5.loss(y),updates=updates)
    leNet5_error=function([x,y],leNet5.error(y))
    leNet5_pred=function([x],leNet5.pred)
    for i in range(itr):
        print i
        train_loss=leNet5_model(trainx,trainy)
    train_error=leNet5_error(validx,validy)
    train_pred=leNet5_pred(testx)
    print "predict and true class"
    for i in xrange(testy.shape[0]):
        print train_pred[i],testy[i]

    print "train_error"
    print train_error
    print "test error"
    test_error=(train_pred!=testy).mean()
    print test_error


if __name__ == '__main__':
    run_test()


