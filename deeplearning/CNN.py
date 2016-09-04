#!/usr/bin/python
#Filename:CNN.py
#Function:CNN layer
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-03

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared
from theano.tensor.signal.pool import pool_2d

class CNN:
    def __init__(self,x,image_shape,filter_shape,poolsize=(2,2)):
        #randomly initial W and b 
        print image_shape[1]
        print filter_shape[1]
        #assert image_shape[1]==filter_shape[1] #channal shouble be equal
        factor = np.prod(filter_shape[1:])
        low=-np.sqrt(1./np.sqrt(factor))
        high=np.sqrt(1./np.sqrt(factor))
        random_w = np.random.uniform(low=low,high=high,
            size=filter_shape)
        #print random_w
        self.W = shared(np.asarray(random_w,dtype=theano.config.floatX))
        self.b = shared(np.zeros(filter_shape[0]
            ,dtype=theano.config.floatX))
        self.params=[self.W,self.b]
        conv_out=T.nnet.conv2d(x,self.W)
        pool_out=pool_2d(conv_out,poolsize,ignore_border=True)
        self.output=T.nnet.sigmoid(pool_out+self.b.dimshuffle('x',0,'x','x'))
        #print self.b.get_value()

#test CNN and maxpooling function
def run_test(datafile='data/3wolfmoon.jpg'):
    from PIL import Image
    import matplotlib.pyplot as plt
    img=np.asarray(Image.open(datafile))/256.
    img=img.astype('float32')
    print img.shape
    fine_img=img.swapaxes(0,2).swapaxes(1,2).reshape(1,3,
            img.shape[0],img.shape[1])
    x=T.ftensor4('x')
    filter_shape=(5,3,9,9)
    cnn = CNN(x,fine_img.shape,filter_shape)
    f=function([x],cnn.output)
    filter_img=f(fine_img)
    print "filter_img shape"
    print filter_img.shape
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.gray()
    plt.subplot(2,3,2)
    plt.imshow(filter_img[0,0,:,:])
    plt.subplot(2,3,3)
    plt.imshow(filter_img[0,1,:,:])
    plt.subplot(2,3,4)
    plt.imshow(filter_img[0,2,:,:])
    print filter_img[0,1,1,1]
    plt.subplot(2,3,5)
    plt.imshow(filter_img[0,3,:,:])
    plt.subplot(2,3,6)
    plt.imshow(filter_img[0,4,:,:])
    plt.show()

if __name__ == '__main__':
    run_test()

