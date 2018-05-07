# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:58:04 2017

@author: Common
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 01:04:27 2017

@author: PhantomV
my

-to convert Image.open() obj to array, use np.array(obj)
-to convert array to img, use Image.fromarray(np.uint8((obj)*255)).show()
-Output size: (W−F+2P)/S+1
Example: 7x7 image, 3x3 filter, 2x2 stride, output size=(7-3+2(0))/2+1=3 on each side (or 3x3 matrix)

-field size very import wtf image size

"""


import numpy as np
import pdb
import tensorflow as tf
import data_handler as dh
from PIL import Image, ImageOps

qwe=pdb.set_trace







class ConvolutionalNeuralNetwork():
    def __init__(self,
                 lr=0.02,#lr=0.001,             
                 field_size=[6,6],#<----field size does not affect output size
                 cv_sizes=[32,64,128],
                 fc_sizes=[100],
                 strides=[1,2,2,1],#<----affects output size
                 pool_size=[1,2,2,1],
                 pool_strides=[1,2,2,1],      #<----affects output size

                 dropout=1                          
                ):        
        self.lr=lr      
        self.field_size=field_size
        self.cv_sizes=cv_sizes
        self.fc_sizes=fc_sizes
        self.strides=strides
        self.pool_size=pool_size
        self.pool_strides=pool_strides      
        self.dropout=dropout
        self.sess = tf.InteractiveSession()#<--------------------tf session is opened here
                                         
        self.record=[]


    
    def train(self,trainset,Y_labels,epoch,output=True):   
        self.input_size=np.shape(trainset[0])
        self.classes_no=np.shape(Y_labels[0])[0]       
        ''' weight & biases'''
        self.init_cv_weights(trainset)
        self.init_fc_weights()
        self.init_bias()
        
        
        '''input'''
        self.X = tf.placeholder("float", [None, self.input_size[0],self.input_size[1],self.input_size[2]])
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
    
    
    
    def train2(self,trainset,Y_labels,output=True):   
        self.input_size=np.shape(trainset[0])
        self.classes_no=np.shape(Y_labels[0])[0]       
        ''' weight & biases'''
        self.init_cv_weights(trainset)
        self.init_fc_weights()
        self.init_bias()
        
        
        '''input'''
        self.X = tf.placeholder("float", [None, self.input_size[0],self.input_size[1],self.input_size[2]])
        self.Y = tf.placeholder("float", [None, self.classes_no])
        
        
        pred_y=self.fprop(self.X)     
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=self.Y)) # compute costs
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost) # construct an optimizer       
        
        
        #start of tf initializer!!        
        tf.global_variables_initializer().run() 
        lastR=999       
        while(True):  
            self.sess.run(train_op, feed_dict={self.X:trainset,self.Y:Y_labels})
            cos=self.sess.run(cost,feed_dict={self.X:trainset,self.Y:Y_labels})
            if(output):print(cos)  
            if cos>=lastR:
                break
            else:
                lastR=cos
        return

    def use(self,useset):
        predict_op = tf.argmax(self.fprop(useset), 1)        
        return self.sess.run(predict_op)
    
    def accuracy(self,testset,y):          
        correct_prediction = tf.equal(tf.argmax(y, 1), self.use(testset)) 
        return  self.sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))   
    
    def fprop(self,x):
        x=self.model(x,False)
        return x
            
    
    
    def init_cv_weights(self,x):
        self.cv_weights,sizes=[],self.cv_sizes        
        for layer in range(len(sizes)):            
            prevSize=self.input_size[2]#<we are taking depth dimension
            if layer>0:
                prevSize=sizes[layer-1]       
            #w = tf.Variable(tf.random_normal([3, 3, 3, 3], stddev=0.01))  
            self.cv_weights.append(tf.Variable(tf.random_normal([self.field_size[0],self.field_size[1],prevSize,sizes[layer]], stddev=0.01)))
        
        for i in range(len(self.cv_weights)):#<----this part for creating dimen for fc
            x=self.model_cv(x,i)       
        self.fc_input_size=int(tf.reshape(x[0],[-1,1]).shape[0])
        return            
                
    
    
    def init_fc_weights(self):
        #use after cv_w initialization
        self.fc_weights,sizes=[],self.fc_sizes    
        for layer in range(len(sizes)):            
            prevSize=self.fc_input_size
            if layer>0:
                prevSize=sizes[layer-1]  
            self.fc_weights.append(tf.Variable(tf.random_normal([prevSize,sizes[layer]], stddev=0.01)))
        self.fc_weights.append(tf.Variable(tf.random_normal([sizes[layer],self.classes_no], stddev=0.01)))#last layer is fully connected layer
        return
        
    def init_bias(self):
        self.biases,sizes=[],self.cv_sizes     
        for layer in range(len(sizes)):           
            self.biases.append(tf.Variable(tf.random_normal([sizes[layer]], stddev=0.01)))
        sizes=self.fc_sizes   
        for layer in range(len(sizes)):           
            self.biases.append(tf.Variable(tf.random_normal([sizes[layer]], stddev=0.01)))            
        self.biases.append(tf.Variable(tf.random_normal([self.classes_no], stddev=0.01)))
        return  
    
    

    def model(self,x,storeResult):
        '''
        def model(self,input,i):  
        h=tf.nn.sigmoid(tf.matmul(input, self.weights[i])+self.biases[i])        
        return h
        '''
        self.storedResult=[]        
        
        for i in range(len(self.cv_sizes)):                
            x=tf.nn.dropout(x,self.dropout)
            x=tf.nn.conv2d(x, self.cv_weights[i],self.strides, padding='SAME')
            x=tf.nn.relu(x)        
            x=tf.nn.max_pool(x,self.pool_size,self.pool_strides, padding='SAME')
            if(storeResult):
                self.storedResult.append(x)
        
        x=tf.reshape(x,[-1,self.fc_input_size])
        for j in range(len(self.fc_sizes)):
            x=tf.nn.dropout(x,self.dropout)     
            x=tf.matmul(x,self.fc_weights[j])
            x=tf.nn.relu(x)         
        x=tf.matmul(x,self.fc_weights[-1])
        
        def a():
            return 0
        return x
    
    def model_cv(self,x,i):
        '''
        def model(self,input,i):  
        h=tf.nn.sigmoid(tf.matmul(input, self.weights[i])+self.biases[i])        
        return h
        '''
        x=tf.nn.dropout(x,self.dropout)
        h0=tf.nn.conv2d(x, self.cv_weights[i],self.strides, padding='SAME')
        h1=tf.nn.relu(h0)        
        h2=tf.nn.max_pool(h1,self.pool_size,self.pool_strides, padding='SAME')          
        return h2
    
    
     


x,y=dh.createDatasets('imageSet//trainset',size=[100,100])
xt,yt=dh.createDatasets('imageSet//testset',size=[100,100])

cnn=ConvolutionalNeuralNetwork()
cnn.train(x,y,100)
#cnn.train2(x,y)
#cnn.see_FC()
print("Accuracy:",cnn.accuracy(x,y))
test=cnn.use(xt)
for i in test:
    print(i)
dh.csvOutput(cnn.use(x),'output//output_cnn.csv')    

cnn.model(x,True)
sr=cnn.storedResult
#Image.fromarray(cnn.sess.run(sr[0][0][:,:,2:5])*255).show()
def q(layer,dataNo,i,threshold):
    for i in range(i):
        im=cnn.sess.run(sr[layer][dataNo][:,:,i:i+3])
        if(np.mean(im)>threshold):
            print(np.mean(im))
            Image.fromarray(np.uint8(im)*255).show()
#Image.fromarray(np.uint8(cnn.sess.run(sr[0][0][:,:,2:5])*255)).show()
#q(0,4,30,0.7)