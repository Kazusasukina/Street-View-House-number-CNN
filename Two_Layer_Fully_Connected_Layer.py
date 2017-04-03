#!/usr/bin/env python
#coding:utf-8
"""
  Author:  weiyiliu --<weiyiliu@us.ibm.com>
  Purpose: 采用一个全连接的网络 [input_layer]->[hidden_layer]->[output_layer]
  Created: 04/02/17
"""
import time

import load_Datasets as LD
import tensorflow as tf
import numpy as np

from Popular_NN_functions import get_chunk_iterator

########################################################################
class FC_Network:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, num_hidden, \
                 batch_size,image_size,num_channel,num_labels, \
                 train_sample,train_label,test_sample,test_label):
        """
        @num_hidden : 隐藏层数
        
        @batch_size : 每次输入的图片个数
        @image_size : 图片大小
        @num_channel: 通道数(一张图片默认有 R+G+B 三通道哈)
        @num_label  : 标签多少(一张图片的标签有0~9共10个标签)
        
        @train_sample : 训练数据
        @train_label  : 训练标签
        @test_sample  : 验证数据
        @test_label   : 验证标签
        """
        # 1. 和TF的图有关
        self.graph = tf.Graph()
        self.num_hidden = num_hidden
        
        # 2. 训练数据有关
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channel = num_channel
        self.num_labels = num_labels
       
        #self.tf_train_samples = None
        #self.tf_train_labels = None
        #self.tf_test_samples = None
        #self.tf_test_labels = None
        
    #----------------------------------------------------------------------
    def graphConstruction(self):
        """
        定义计算图谱
        """
        with self.graph.as_default():
            """
            Train/Test所用的 图片
            -------------
            shape形状:[batch_size, image_size,image_size, num_channel]
                1. batch_size            # 每次读入的图片数量
                2. image_size,image_size # 当前图片大小(长*宽)
                3. num_channel           # 通道个数:
                                           注意一下，因为之前在LOAD里面我们已经将RGB映射成了灰度图，因此在这里的num_channle=1
            -----------------------------------------------------------------
            Train/Test所用的 标签
            -------------
            shape形状:[batch_size, num_label]
                1. batch_size            # 每次读入的图片数量
                2. num_label             # 标签one-hot-vector
            -----------------------------------------------------------------
            """
            
            self.tf_train_samples = tf.placeholder(tf.float32,shape=[self.batch_size, 
                                                                     self.image_size, 
                                                                     self.image_size,
                                                                     self.num_channel]
                                                   )
            self.tf_test_samples = tf.placeholder(tf.float32,shape=[self.batch_size,\
                                                                     self.image_size,\
                                                                     self.image_size,\
                                                                     self.num_channel]\
                                                   )
            self.tf_train_labels = tf.placeholder(tf.float32,shape=[self.batch_size, self.num_labels])
            
            """
            定义全连接的Fully Connected NN
            
            第一层为隐藏层(因为输入层没有必要当成一层哈)
            -------------
            Weight Shape形状: [input的图片, input的图片的label]
                1. input的图片      # 每次input为 1 张 imageSize*imageSize*num_channel 的图片 
                                     [1张网络是因为每次神经元都只会学一张image，而不是学Batch_size张image!这里需要特别注意!!]
                2. input的图片label # 这个图片是哪一个类的
            Bias Shape形状: [num_hidden] # 隐藏层的神经元个数
            -----------------------------------------------------------------
            第二层为输出层
            -------------
            Weight Shape形状: [num_hidden, num_label]
                1. input应该是隐含层的输出 # 这里需要注意的是，神经网络的上一层的输出应该是下一层的输入！这点尤为重要！
                2. num_label            # 输出的应该是One-Hot Vector 代表的是 0~9 中的数字几
            Bias Shape 形状: [num_label] # 输出是一个One-Hot Vector 代表的是 0~9 中的数字几
            -----------------------------------------------------------------
            """
            self.fc1_weights = tf.Variable(tf.truncated_normal([self.image_size*self.image_size*self.num_channel,
                                                           self.num_hidden],stddev=0.1)
                                      )
            self.fc1_bias    = tf.Variable(tf.constant(0.1,shape=[self.num_hidden]))
            
            self.fc2_weights = tf.Variable(tf.truncated_normal([self.num_hidden,self.num_labels],stddev=0.1))
            self.fc2_bias    = tf.Variable(tf.constant(0.1,shape=[self.num_labels]))
            
            """
            计算全连接层
            """
            # 习惯把最后一个全连接的输出称为 logits
            self.logits = self.__model(self.tf_train_samples, 
                                       self.fc1_weights,self.fc1_bias,self.fc2_weights,self.fc2_bias)

            # 定义loss function 为 cross entropy 最小
            # 这里 真实 的数据分布为 self.tf_train_labels
            #     预测 的数据分布为 logits 即output-layer的输出
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                               labels=self.tf_train_labels))
            #self.loss = -tf.reduce_mean(self.logits*tf.log(self.tf_train_labels),reduction_indices=[1])

            # 优化方式为 loss function 最小
            self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
            
            # 最后接上softMax判断当前输出 
            self.train_prediction = tf.nn.softmax(self.logits)
            
    
    #----------------------------------------------------------------------
    def run(self, samples_datasets='',labels_datasets='',chunk_size=100):
        """
        创建Session, 运行该Fully_Connected_NN
        如果需要每次batch的进行运行,则还需设置input参数
        @chunk_size = 100               : 默认每次读入的chunk大小为100
        @samples_datasets    -> Samples Datasets : 需指定 sample 数据集
        @labels_datasets     -> Labels Datasets  : 需指定 label 数据集
        """
        #self.session = tf.Session(graph=self.graph)
        self.session = tf.InteractiveSession(graph=self.graph)
        #with self.session as session: # 这样会在后面还需要出现Attempted to use a closed Session.问题
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        print('Start Training Process...')
        for i, sample, label in get_chunk_iterator(samples_datasets, labels_datasets, chunk_size=self.batch_size):
            _, currentLoss, predictions = self.session.run([self.optimizer, self.loss, self.train_prediction],
                                                      feed_dict={self.tf_train_samples : sample,
                                                                 self.tf_train_labels  : label}
                                                      )
            if i%100 == 0:
                print('current loss: ', currentLoss, 
                      '\tcurrent prediction: ', self.__accuracy(predictions, label), '%')

    #----------------------------------------------------------------------
    def test_accuracy(self, samples_datasets='',labels_datasets='',chunk_size=100):
        """
        测试模型的正确率
        """
        sampleData = samples_datasets
        labelDatasets = labels_datasets
        
        self.tf_test_labels = tf.placeholder(tf.float32,shape=[self.batch_size, self.num_labels])

        self.test_prediction = tf.nn.softmax(self.__model(self.tf_test_samples,
                                                          self.fc1_weights,
                                                          self.fc1_bias,
                                                          self.fc2_weights,
                                                          self.fc2_bias
                                                          )
                                             )

        accuracies = []
        for i, samples, labels in get_chunk_iterator(sampleData,labelDatasets):
            result = self.test_prediction.eval(session=self.session, feed_dict={self.tf_test_samples:samples})
            accuracy = self.__accuracy(result, labels)
            accuracies.append(accuracy)
            print('Test Accuracy: ',accuracy)
        print('Average Accuracy: ', np.average(accuracies))

    #----------------------------------------------------------------------
    def __model(self,train_samples, fc1_w, fc1_b, fc2_w, fc2_b):
        """
        定义模型
        @train_samples: 输入图像
        @fc1_w        : 隐藏层的权重
        @fc1_b        : 隐藏层的bias
        @fc2_w        : 输出层的权重
        @fc2_b        : 输出层的bias
        """
        shape = train_samples.get_shape().as_list()
        reshape = tf.reshape(train_samples, [shape[0],shape[1]*shape[2]*shape[3]])
        hidden_layer = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)
        """
        output_layer = tf.nn.relu(tf.matmul(hidden_layer, fc2_w) + fc2_b)
        注意: 输出层不加relu激活函数的原因是因为激活函数relu其实会选特征，
             这些对特征的“选择”有可能会导致结果的不准确性
             所以在输出层就直接将结果相加，而不能利用激活函数进行过滤
        """
        output_layer = tf.matmul(hidden_layer, fc2_w) + fc2_b
        
        return output_layer
    
    #----------------------------------------------------------------------
    def __accuracy(self, prediction, labels, ):
        """
        计算正确率
        """
        _prediction = np.argmax(prediction, 1)
        _labels = np.argmax(labels, 1)
        accuracy = np.sum(_prediction == _labels) / len(prediction)
        return 100*accuracy


if __name__ == "__main__":
    start = time.time()
    # 1. 设置图片数据格式并导入数据
    imageSize,num_channel,num_labels = LD.SetImageProperty()
    train_path = 'SVHN_datas/train_32x32.mat'
    test_path  = 'SVHN_datas/test_32x32.mat'
    train_sample, train_label, test_sample, test_label = LD.loadDatasets(train_path,test_path)

    # 2. 初始化网络 ---> 这里设置隐藏层中的神经元个数为128个
    num_hidden = 128
    batch_size = 100
    FC = FC_Network(num_hidden, batch_size, imageSize, num_channel, num_labels,\
                    train_sample, train_label, test_sample, test_label)

    # 3. 构建网络
    FC.graphConstruction()

    # 4. RUN Model
    FC.run(samples_datasets=train_sample,labels_datasets=train_label)
    
    FC.test_accuracy(samples_datasets=test_sample,labels_datasets=test_label)
    
    print('down!', 'Total Time is: ', time.time()-start)