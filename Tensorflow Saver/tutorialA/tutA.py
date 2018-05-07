# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:05:41 2017

@author: Common
"""

import tensorflow as tf
import pdb 
import os
qwe=pdb.set_trace

def a(x):
    #Prepare to feed input, i.e. feed_dict and placeholders
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1= tf.Variable(2.0,name="bb")
    feed_dict ={w1:4,w2:8}
    
    #Define a test operation that we will restore
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore")
    sess = tf.Session()
    qwe()
    if(x==1):
        #tf.reset_default_graph()   #sess.run('bb:0')
        #sess2=tf.Session()
        saver = tf.train.import_meta_graph(os.getcwd()+'\\xxxx.meta') 
        saver.restore(sess,tf.train.latest_checkpoint('./'))
    else:
        sess.run(tf.global_variables_initializer())    
    #Create a saver object which will save all the variables
    saver = tf.train.Saver()
    
    #Run the operation by feeding input
    print (sess.run(w4,feed_dict))
    #Prints 24 which is sum of (w1+w2)*b1 
    
    #Now, save the graph
    saver.save(sess, os.getcwd()+'\\xxxx')


def b():    
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.getcwd()+'\\xxxx.meta') 
    saver.restore(sess,tf.train.latest_checkpoint('./'))   
    # Access saved Variables directly
    print(sess.run('bb:0'))

def c():
    w1 = tf.placeholder("float", name="w1")
    qwe()
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(os.getcwd()+'\\xxxx.meta') 
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    
    # Access saved Variables directly
    print(sess.run('bb:0'))