#!/bin/python
#Filename:LogisticRegression.py
#Function:biclassifer using LR
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-01
import numpy as np
import theano.tensor as T
from theano import In
from theano import function
from theano import shared

rng=np.random
N=400
in_n=784

#generate point(data,label)
D=(rng.rand(N,in_n),rng.randint(0,2,N))
print "data"
print D

#generate model
x=T.dmatrix('x')
y=T.dvector('y')

#set default w and b
W=shared(rng.randn(in_n))
b=shared(0.0)

print "Initial model"
print W.get_value()
print b.get_value()

hx=1/(1+T.exp(-T.dot(x,W)-b))
prediction = hx>0.5
xcent = -y*T.log(hx)-(1-y)*T.log(1-hx)
cost = xcent.mean()+0.01*(W**2).sum()
gw,gb=T.grad(cost,[W,b])

train=function([x,y],[prediction,xcent],updates=[(W,W-0.1*gw),(b,b-0.1*gb)])
for i in range(10000):
    print i
    pred,err = train(D[0],D[1])

print "final model"
print W.get_value()
print b.get_value()

target=D[1]
print "target values and predction for D:"
for i in range(N):
    print target[i],pred[i]
print "prediction error:"
print (target!=pred).mean()
    

