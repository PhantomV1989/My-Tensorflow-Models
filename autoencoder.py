# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:41:01 2017

@author: Common
"""

import tensorflow as tf
import numpy as np
import pdb
import data_handler as dh

qwe=pdb.set_trace

class Autoencoder():
        def __init__(self,
                 lr=0.9,            
                 sizes=[10],
                 dropout=1                                             
                ):              
            self.lr=lr       
            self.sizes=sizes
            self.dropout=dropout        
            self.sess = tf.InteractiveSession()#<--------------------tf session is opened here
        
        def train(self,x,epoch,output=True): 
            self.input_size=len(x[0])
   
            ''' weight & biases'''
            self.init_w()
            self.init_b()
            
            
            '''input'''
            self.X = tf.placeholder("float", [None, self.input_size])
            
            pred_x=self.model(self.X)     
            self.cost = tf.reduce_mean(tf.pow(self.X - pred_x, 2))  # minimize squared error            
            train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost) # construct an optimizer       
            
            
            #start of tf initializer!!
          
            tf.global_variables_initializer().run() 
           
            for i in range(epoch):  
                self.sess.run(train_op, feed_dict={self.X:x})
                if(output):print(self.sess.run(self.cost,feed_dict={self.X:x}))      
            return
        
        
        def init_w(self):
            self.w,self.wp,sizes=[],[],self.sizes
            for layer in range(len(sizes)):
                prevSize=self.input_size
                if layer>0:
                    prevSize=sizes[layer-1]       
                self.w.append(tf.Variable(tf.random_normal([prevSize,sizes[layer]], stddev=0.01)))
                self.wp.append(tf.transpose(self.w[layer]))
            #self.w.append(tf.Variable(tf.random_normal([sizes[layer],self.classes_no], stddev=0.01)))#add last layer for classification
            return  
        
        def init_b(self):
            self.b,sizes=[],self.sizes
            for layer in range(len(sizes)):           
                self.b.append(tf.Variable(tf.zeros([sizes[layer]])))
            #self.b.append(tf.Variable(tf.random_normal([self.classes_no], stddev=0.01)))
            return
        '''
        def init_c(self):
            self.c,sizes=[],self.sizes
            for i in range(len)
        '''
        
        def model(self,x): 
            x=tf.nn.dropout(x,self.dropout)
            for i in range(len(self.w)):#normal fprop
                x=tf.matmul(x,self.w[i])#+self.b[i]
                x=tf.nn.sigmoid(x)  
            for ii in range(len(self.w)-1,-1,-1):#reverse prop
                x=tf.matmul(x,self.wp[ii])#+self.c[j]
                x=tf.nn.sigmoid(x)    
            return x
        
        def use(self,x):                    
            pred_x=self.sess.run(self.model(x))
            for i in range(len(pred_x)):
                costX=self.sess.run(tf.reduce_mean(tf.pow(x[i] - pred_x[i], 2)))
                print("Pos: ",i,"  Cost:",costX)
            return pred_x



def testA():
    #this part shows the abiliy of autoencoders to recognise piecewise patterns 
    x,y=dh.createSetFromCSV('dataset//ac1.csv')
    xt,yt=dh.createSetFromCSV('dataset//ac2.csv')
    x=np.array(x,np.float32)
    ae=Autoencoder(sizes=[30,20],lr=0.1)
    ae.train(x,1000,output=True)
    print("Showing results for original set:")
    ae.use(x)
    print("Showing results for test set:")
    ae.use(xt)
