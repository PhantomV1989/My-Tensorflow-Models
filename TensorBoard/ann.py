# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:56:11 2017

@author: Common

#always clear tensorboard directory before using new graph
1) By plotting histogram weights, can see vanishing gradients (last layer weight got changes vs epoch, 1st layer no change)
2) Viewing browser results while training can see results in "semi" real time
"""
# !/usr/bin/env python




# command prompt
# tensorboard --logdir=C:\Users\PhantomV\Documents\GitHub - Copy\Machine-Learning-Tools\Tensorflow\TensorBoard\logs\nn_logs

# use browser:
# http://localhost:6006/

# tf.name_scope clusters elements in the graph according to their name scope (neater)


import tensorflow as tf
import numpy as np
import pdb
import data_handler as dh
import os

qwe = pdb.set_trace


class NeuralNetwork():
    def __init__(self,
                 lr=0.02,  # lr=0.001,
                 dc=1e-10,
                 sizes=[5],  # sizes=[200,100,50],
                 L2=0.001,
                 L1=0
                 ):
        self.lr = lr
        self.dc = dc
        self.sizes = sizes
        self.L2 = L2
        self.L1 = L1

        self.saverName = "ann"
        for layer in range(len(sizes)):
            self.saverName += ('_' + str(sizes[layer]))
        self.saverDir = os.getcwd() + '\\saved\\'
        if (os.path.exists(self.saverDir) != True):
            os.makedirs(self.saverDir)

    def init_weights(self):
        self.weights, sizes = [], self.sizes
        for layer in range(len(sizes)):
            prevSize = self.input_size
            if layer > 0:
                prevSize = sizes[layer - 1]
            self.weights.append(
                tf.Variable(tf.random_normal([prevSize, sizes[layer]], stddev=0.01), name=('w' + str(layer))))
            tf.summary.histogram('w' + str(layer), self.weights[layer])
        self.weights.append(tf.Variable(tf.random_normal([sizes[layer], self.classes_no], stddev=0.01),
                                        name=('w' + str(layer + 1))))  # add last layer for classification
        tf.summary.histogram('w' + str(layer + 1), self.weights[layer + 1])
        return

    def init_bias(self):
        self.biases, sizes = [], self.sizes
        for layer in range(len(sizes)):
            self.biases.append(tf.Variable(tf.random_normal([sizes[layer]], stddev=0.01), name='b' + str(layer)))
            tf.summary.histogram('b' + str(layer), self.weights[layer])
        self.biases.append(tf.Variable(tf.random_normal([self.classes_no], stddev=0.01), name='b' + str(layer + 1)))
        tf.summary.histogram('b', self.weights[layer])
        return

    def train(self, trainset, Y_labels, epoch, output=True, save=True, load=False, printInterval=1000):

        # start of tf initializer!!

        #       sess.run('w0:0')
        # print(sess.run(self.weights[0]))

        sess = tf.Session()


        if (load & (os.path.exists(self.saverDir + self.saverName + '.meta'))):
            ns = tf.train.import_meta_graph(self.saverDir + self.saverName + '.meta')
            ns.restore(sess, tf.train.latest_checkpoint(self.saverDir))  # restore Variables
        else:
            if (load): print('Load file not found.')
            self.input_size = len(trainset[0])
            self.classes_no = len(Y_labels[0])

            ''' weight & biases'''
            self.init_weights()
            self.init_bias()

            '''input'''
            with tf.name_scope('input'):
                self.X = tf.placeholder("float", [None, self.input_size], name="X")
                self.Y = tf.placeholder("float", [None, self.classes_no], name="Y")
            pred_y = self.fprop(self.X)

            with tf.name_scope("cost"):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=self.Y),
                                      name='cost')  # compute costs
                train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost,
                                                                                             name='train')  # construct an optimizer
                tf.summary.scalar("cost", cost)

            sess.run(tf.global_variables_initializer())

            '''writer'''
            # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
        writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)  # for 1.0
        merged = tf.summary.merge_all()

        for i in range(epoch):
            _, summary = sess.run(['cost/train', merged], feed_dict={'input/X:0': trainset, 'input/Y:0': Y_labels})
            writer.add_summary(summary, i)
            if (output & (i % printInterval == 0)):
                print(sess.run('cost/cost:0', feed_dict={'input/X:0': trainset, 'input/Y:0': Y_labels}))

        if (save):
            saver = tf.train.Saver()
            saver.save(sess, self.saverDir + self.saverName)
        tf.reset_default_graph()
        sess.close()
        return

    def model(self, input, i):
        h = tf.nn.relu(tf.matmul(input, self.weights[i]) + self.biases[i])
        return h

    def fprop(self, input_d):
        for layer in range(len(self.weights)):
            input_d = self.model(input_d, layer)
        return input_d

    def use(self, useset):
        predict_op = tf.argmax(self.fprop(useset), 1)
        return self.sess.run(predict_op)

    def accuracy(self, testset, y):
        correct_prediction = tf.equal(tf.argmax(y, 1), self.use(testset))
        return self.sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))


if __name__ == "__main__":
    x, y = dh.createSetFromCSV('training.csv')
    testset, y_ = dh.createSetFromCSV('test.csv')
    ann = NeuralNetwork(sizes=[500], lr=0.01)
    ann.train(x, y, 1500, printInterval=100, output=True, save=False, load=False)
    # print("Accuracy:",ann.accuracy(testset,y_))
    # dh.csvOutput(ann.use(testset),'output//output_ann.csv')
