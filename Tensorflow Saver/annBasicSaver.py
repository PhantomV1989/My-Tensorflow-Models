# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:56:11 2017

@author: Common
"""
#if file load problem, try saving in same folder as py file
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pdb
import data_handler as dh
import os


qwe=pdb.set_trace
class NeuralNetwork():
    def __init__(self,
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
        self.saverName="ann"
        for layer in range(len(sizes)):
            self.saverName+=('_'+str(sizes[layer]))
        self.saverDir=os.getcwd()+'\\saved\\'
        if(os.path.exists(self.saverDir)!=True):
            os.makedirs(self.saverDir)
            
            
            
        
    def init_weights(self):        
        self.weights,sizes=[],self.sizes    
        for layer in range(len(sizes)):            
            prevSize=self.input_size
            if layer>0:
                prevSize=sizes[layer-1]               
            self.weights.append(tf.Variable(tf.random_normal([prevSize,sizes[layer]], stddev=0.01),name=('w'+str(layer))))
        self.weights.append(tf.Variable(tf.random_normal([sizes[layer],self.classes_no], stddev=0.01),name=('w'+str(layer+1))))#add last layer for classification
        return            
                
    def init_bias(self):
        self.biases,sizes=[],self.sizes
        for layer in range(len(sizes)):           
            self.biases.append(tf.Variable(tf.random_normal([sizes[layer]], stddev=0.01),name='b'+str(layer)))
        self.biases.append(tf.Variable(tf.random_normal([self.classes_no], stddev=0.01),name='b'+str(layer+1)))
        return
    

   
    
    
    def train(self,trainset,Y_labels,epoch,output=True,save=True,load=False,printInterval=1000):          
        
        #start of tf initializer!!
       
        #       sess.run('w0:0')
        #print(sess.run(self.weights[0]))
        sess=tf.Session()    
        if(load &(os.path.exists(self.saverDir+self.saverName+'.meta'))):  
            ns = tf.train.import_meta_graph(self.saverDir+self.saverName+'.meta') 
            ns.restore(sess, tf.train.latest_checkpoint(self.saverDir)) #restore Variables
        else:
            if(load):print('Load file not found.')
            self.input_size=len(trainset[0])
            self.classes_no=len(Y_labels[0])
            
            ''' weight & biases'''
            self.init_weights()
            self.init_bias()
            
            
            '''input'''
            self.X = tf.placeholder("float", [None, self.input_size],name="X")
            self.Y = tf.placeholder("float", [None, self.classes_no],name="Y")                 
            pred_y=self.fprop(self.X)     
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=self.Y),name='cost') # compute costs
            train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost,name='train') # construct an optimizer       
            sess.run(tf.global_variables_initializer())        
            
       
        for i in range(epoch):              
            sess.run('train', feed_dict={'X:0':trainset,'Y:0':Y_labels})
            if(output & (i%printInterval==0)):                
                print(sess.run('cost:0',feed_dict={'X:0':trainset,'Y:0':Y_labels}))   
              
        
        
        if (save):     
            saver=tf.train.Saver()
            saver.save(sess, self.saverDir+self.saverName)
        tf.reset_default_graph()
        sess.close()
        return
    
    def model(self,input,i):  
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
    

        

if __name__=="__main__":     
    x,y=dh.createSetFromCSV('training.csv')
    testset,y_=dh.createSetFromCSV('test.csv')    
    ann=NeuralNetwork(sizes=[50],lr=0.01)
    ann.train(x,y,1500,printInterval=100,output=True,save=True,load=False)   
    #print("Accuracy:",ann.accuracy(testset,y_))
    #dh.csvOutput(ann.use(testset),'output//output_ann.csv')    
