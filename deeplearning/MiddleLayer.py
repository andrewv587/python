#!/usr/bin/python
#Filename:MiddleLayer.py
#Function:the middle layer of Multi-perception
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-03

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

class MiddleLayer:
    # model:y=tanh(x*W+b)
    # x:input parameters(n*n_in)
    # y:output parameters(n*n_out()
    # W(n_in*n_out) b(n_out):model parameters
    def __init__(self,x,n_in,n_out):
        #randomly initial W and b 
        low=-np.sqrt(6./(n_in+n_out))
        high=np.sqrt(6./(n_in+n_out))
        random_w = np.random.uniform(low=low,high=high,
            size=(n_in,n_out))
        #print random_w
        self.W = shared(np.asarray(random_w,dtype=theano.config.floatX))
        self.b = shared(np.zeros(n_out
            ,dtype=theano.config.floatX))
        self.y=T.tanh(T.dot(x,self.W)+self.b)
        #print self.b.get_value()

