# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:56:11 2017

@author: Common
"""

#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pdb
import data_handler as dh


qwe=pdb.set_trace
class NeuralNetwork():
    def __init__(self,
                 dropout=0.5,
                 lr=0.02,#lr=0.001,
                 dc=1e-10,
                 sizes=[5],#sizes=[200,100,50],
                 L2=0.001,
                 L1=0                             
                ):        
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1     
        self.dropout=dropout
        self.sess = tf.InteractiveSession()#<--------------------tf session is opened here
        
        
    def init_weights(self):        
        self.weights,sizes=[],self.sizes    
        for layer in range(len(sizes)):
            prevSize=self.input_size
            if layer>0:
                prevSize=sizes[layer-1]           
            self.weights.append(tf.Variable(tf.random_normal([prevSize,sizes[layer]], stddev=0.01)))
        self.weights.append(tf.Variable(tf.random_normal([sizes[layer],self.classes_no], stddev=0.01)))#add last layer for classification
        return            
                
    def init_bias(self):
        self.biases,sizes=[],self.sizes
        for layer in range(len(sizes)):           
            self.biases.append(tf.Variable(tf.random_normal([sizes[layer]], stddev=0.01)))
        self.biases.append(tf.Variable(tf.random_normal([self.classes_no], stddev=0.01)))
        return
    

    
    def train(self,trainset,Y_labels,epoch,output=True):   
        self.input_size=len(trainset[0])
        self.classes_no=len(Y_labels[0])
        
        ''' weight & biases'''
        self.init_weights()
        self.init_bias()
        
        
        '''input'''
        self.X = tf.placeholder("float", [None, self.input_size])
        self.Y = tf.placeholder("float", [None, self.classes_no])
        
        
        pred_y=self.fprop(self.X)     
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=self.Y)) # compute costs
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost) # construct an optimizer       
        
        
        #start of tf initializer!!
      
        tf.global_variables_initializer().run() 
       
        for i in range(epoch):  
            self.sess.run(train_op, feed_dict={self.X:trainset,self.Y:Y_labels})
            if(output):print(self.sess.run(cost,feed_dict={self.X:trainset,self.Y:Y_labels}))
          
        
        return
    
    def model(self,input,i):  
        input = tf.nn.dropout(input, self.dropout)
        h=tf.nn.sigmoid(tf.matmul(input, self.weights[i])+self.biases[i])        
        return h
    
    def fprop(self,input_d):            
        for layer in range(len(self.weights)):
            input_d=self.model(input_d,layer)  
        return input_d
    
    def use(self,useset):
        predict_op = tf.argmax(self.fprop(useset), 1)        
        return self.sess.run(predict_op)
        
    def accuracy(self,testset,y):          
        correct_prediction = tf.equal(tf.argmax(y, 1), self.use(testset)) 
        return  self.sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        

x,y=dh.createSetFromCSV('dataset//training.csv')
testset,y_=dh.createSetFromCSV('dataset//test.csv')    
ann=NeuralNetwork(sizes=[2],lr=0.2,dropout=0.9)
ann.train(x,y,1000,output=False)
print("Accuracy:",ann.accuracy(testset,y_))
dh.csvOutput(ann.use(testset),'outputxxx.csv')    
