#!/usr/bin/env python
#coding:utf-8
"""
  Author:  weiyiliu --<weiyiliu@us.ibm.com>
  Purpose: CNN 
           [input_layer]->[hidden_layer]->[output_layer]
  Created: 04/04/17
"""
import time

import load_Datasets as LD
import tensorflow as tf
import numpy as np

from Popular_NN_functions import get_chunk_iterator

########################################################################
class CNN:
    """"""

    #----------------------------------------------------------------------
    def __init__(self,num_hidden, 
                      batch_size,image_size,num_channel,num_labels, 
                      filter_size,filter_stride,num_filter,pooling_size,pooling_stride
                ):
        """
        @num_hidden    : 隐藏层数
        
        @batch_size    : 每次输入的图片个数
        @image_size    : 图片大小
        @num_channel   : 通道数(一张图片默认有 R+G+B 三通道哈)
        @num_label     : 标签多少(一张图片的标签有0~9共10个标签)
        
        @filter_size   : CONV 中 小滑窗(filter)大小
        @filter_stride : CONV 中 小滑窗的滑动步长
        @num_filter    : CONV 中 小滑窗的个数(即featureMap的个数)
        @pooling_size  : CONV 中 pooling的大小
        @pooling_stride: CONV 中 pooling的滑动步长
        """
        # 1. 和TF的图有关 --- hyper parameters
        self.graph = tf.Graph()
        self.num_hidden = num_hidden
        
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        
        # 卷积层逐层增大
        self.num_filter_layer1 = num_filter
        self.num_filter_layer2 = self.num_filter_layer1 * 2
        self.num_filter_layer3 = self.num_filter_layer2 * 2
        self.num_filter_layer4 = self.num_filter_layer3 * 2

        # 2. 训练数据有关
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channel = num_channel
        self.num_labels = num_labels

        # 3. 构建网络
        self.__graphConstruction()
        self.session = tf.InteractiveSession(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board',self.graph)
        # self.merged = None

    #----------------------------------------------------------------------
    def __graphConstruction(self):
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
            with tf.name_scope('inputs'):
                self.tf_train_samples = tf.placeholder(tf.float32,shape=[self.batch_size, 
                                                                        self.image_size, 
                                                                        self.image_size,
                                                                        self.num_channel], 
                                                                        name="TrainSamples"
                                                    )
                self.tf_train_labels = tf.placeholder(tf.float32,
                                                    shape=[self.batch_size, self.num_labels],
                                                    name="TrainLabels")
            
            """
            定义CNN [input_layer]  -----------------
                                  -> [conv_layer_1]
                                  -> [conv_layer_2]
                                  -> [max_pooling]  # 将高维降成底维
                                  -----------------
                                  -> [conv_layer_3]
                                  -> [conv_layer_4]
                                  -> [max_pooling]  # 将高维降成底维
                                  -----------------
                                  -> [fully_connected_layer_1]
                                  -> [fully_connected_layer_2]
                                  -----------------
                                  -> [soft_max] -> [output_layer]
            
            Conv 卷积层 Detail
            p.s. 和FC不一样，卷积神经网络中每一个卷积层并不需要给出sample的多少，因为这个是由TF自己的函数conv2d控制的!!
            -----------------
            Weight Shape形状: [patch_size, patch_size, 上一层filter个数, 该层filter个数]
                1. 每一个Filter的长*宽*厚度(filter_num) : patch_size,patch_size,num_channel
                                p.s. (第一层CONV由于连接的是图片，所以此时的filter个数就应该是该图片的通道数)
                2. Filter总个数
            Bias Shape形状: [num_filter] # filter 总个数
            -----------------------------------------------------------------

            Fully_Connected 全连接层 Detail
            -------------
            1. FC_layer1 --- 用作对之前卷积神经网络的结果进行处理

                I. 输入的数据: 应该是前面卷积层之后的 batch_size 和 imageSize
                        --- 首先 应该将之前所得到的结果进行扁平化
                            a). 假设第二次 max_pooling 之前 每一个filter的大小为filter_size 同时此时一共有Num_Filter个filter
                            
                            b). 同时max_pooling 的scale 为 pooling_scale, 且一共做了N_times_Pooling
                                此时相当于将原图像下采样了N_times_Pooling次

                            c). 那么 扁平化就应该将该TENSOR拉伸成一个 Vector 且down_scale = pooling_scale ** N_times_Pooling
                                且向量长度应该为 length = (imageSize//down_scale) * (imageSize//down_scale) * Num_Filter

                        --- 之后 将这些结果进行reshape = [shape[0], length] # shape[0] --- batch_size

                II. Weight Shape形状: [length, num_hidden]
                        1. reshape之后的向量长度 length，也代表了所有图像下采样+filter的所有个数
                        2. num_hidden 设置的隐藏层中的神经元 总个数
            Bias Shape形状: [num_hidden] # 该隐藏层中的神经元 总个数

            2. FC_layer2 --- 用于连接最后的softmax层 (该层与全连接神经网络的Output层一致)
            Weight Shape形状: [num_hidden, num_label]
                1. input应该是隐含层的输出 # 这里需要注意的是，神经网络的上一层的输出应该是下一层的输入！这点尤为重要！
                2. num_label            # 输出的应该是One-Hot Vector 代表的是 0~9 中的数字几
            Bias Shape 形状: [num_label] # 输出是一个One-Hot Vector 代表的是 0~9 中的数字几
            -----------------------------------------------------------------
            """
            with tf.name_scope('Conv_1'):
                self.conv1_weight = tf.Variable(tf.truncated_normal([self.filter_size, self.filter_size,
                                                                     self.num_channel,
                                                                     self.num_filter_layer1],
                                                                    stddev=0.1))
                self.conv1_bias = tf.Variable(tf.zeros([self.num_filter_layer1]))

            with tf.name_scope('Conv_2'):
                self.conv2_weight = tf.Variable(tf.truncated_normal([self.filter_size, self.filter_size,
                                                                     self.num_filter_layer1,self.num_filter_layer2],
                                                                    stddev=0.1))
                self.conv2_bias = tf.Variable(tf.constant(0.1,shape=[self.num_filter_layer2]))

            with tf.name_scope('Conv_3'):
                self.conv3_weight = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,
                                                                     self.num_filter_layer2,self.num_filter_layer3],
                                                                    stddev=0.1))
                self.conv3_bias = tf.Variable(tf.constant(0.1, shape=[self.num_filter_layer3]))

            with tf.name_scope('Conv_4'):
                self.conv4_weight = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,
                                                                     self.num_filter_layer3,self.num_filter_layer4],
                                                                    stddev=0.1))
                self.conv4_bias = tf.Variable(tf.constant(0.1, shape=[self.num_filter_layer4]))

            with tf.name_scope('fc1'):
                down_scale = self.pooling_size ** 2 # 一共会进行两次max_pooling，所以相当于imageSize为原来1/4
                self.fc1_weights = tf.Variable(tf.truncated_normal([
                    (self.image_size//down_scale) * (self.image_size//down_scale) * self.num_filter_layer4,
                    self.num_hidden],stddev=0.1), name='hidden_layer_weights'
                )
                self.fc1_bias = tf.Variable(tf.constant(0.1,shape=[self.num_hidden]), name = 'hidden_layer_bias')

            with tf.name_scope('fc2'):
                self.fc2_weights = tf.Variable(tf.truncated_normal([self.num_hidden,self.num_labels],stddev=0.1),name='output_layer_weights')
                self.fc2_bias    = tf.Variable(tf.constant(0.1,shape=[self.num_labels]),name='output_layer_bias')
            
            """
            计算CNN
            """
            # 习惯把最后一个全连接的输出称为 logits
            self.logits = self.__model(self.tf_train_samples) # 构建模型

            # 定义loss function 为 cross entropy 最小
            # 这里 真实 的数据分布为 self.tf_train_labels
            #     预测 的数据分布为 logits 即output-layer的输出
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                               labels=self.tf_train_labels),name='cross_entropy_softmax')
                # self.loss = -tf.reduce_mean(tf.nn.softmax(self.logits*tf.log(self.tf_train_labels)))

            # 优化方式为 loss function 最小
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.001,name="GradientDescent").minimize(self.loss,name='minimize_loss_function')
            
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
        self.session = tf.Session(graph=self.graph)
        init = tf.global_variables_initializer()
        self.session.run(init)

        print('Start Training Process...')
        for i, samples, labels in get_chunk_iterator(samples_datasets, labels_datasets, chunk_size=self.batch_size):
            _, currentLoss, predictions = self.session.run([self.optimizer, self.loss, self.train_prediction],
                                                      feed_dict={self.tf_train_samples : samples,
                                                                 self.tf_train_labels  : labels}
                                                      )
            if i%10 == 0:
                print('current loss: ', currentLoss, 
                      '\tcurrent prediction: ', self.__accuracy(predictions, labels), '%')

    #----------------------------------------------------------------------
    def test_accuracy(self, samples_datasets='',labels_datasets='',chunk_size=100):
        """
        测试模型的正确率
        """
        sampleData = samples_datasets
        labelDatasets = labels_datasets
         
        # 新建两个PlaceHolder分别保存 测试集的真实sample 和 标签信息
        self.tf_test_samples = tf.placeholder(tf.float32, shape=[self.batch_size, 
                                                                  self.image_size, self.image_size,
                                                                  self.num_channel],name="TestSamples")
        self.tf_test_labels = tf.placeholder(tf.float32,shape=[self.batch_size, self.num_labels], name="TestLabels")

        # 利用自己的模型，对测试数据进行分类，之后采用Softmax搞清楚每一个测试样本所属的类别
        self.test_prediction = tf.nn.softmax(self.__model(self.tf_test_samples))

        accuracies = []
        for i, samples, labels in get_chunk_iterator(sampleData,labelDatasets, chunk_size=self.batch_size):
            predict_results = self.test_prediction.eval(session=self.session, feed_dict={self.tf_test_samples:samples})
            accuracy = self.__accuracy(predict_results, labels)
            accuracies.append(accuracy)
            # print('Test Accuracy: ',accuracy)
        print('Average Accuracy: ', np.average(accuracies))

    #----------------------------------------------------------------------
    def __model(self, train_samples):
        """
        定义模型
        @train_samples: 输入图像
        """
        with tf.name_scope('conv1_layer'):
            with tf.name_scope('convolution'):
                conv1 = tf.nn.conv2d(input  = train_samples,
                                     filter = self.conv1_weight,
                                     strides = [1,self.filter_stride,self.filter_stride,1],
                                     padding = "SAME",
                                     use_cudnn_on_gpu=True,
                                     name='conv1')
            conv1_output = tf.nn.relu(conv1+self.conv1_bias)

        with tf.name_scope('conv2_layer'):
            with tf.name_scope('convolution'):
                conv2 = tf.nn.conv2d(input = conv1_output,
                                     filter = self.conv2_weight,
                                     strides = [1,self.filter_stride,self.filter_stride,1],
                                     padding = "SAME",
                                     use_cudnn_on_gpu=True,
                                     name='conv2')
                conv2_output = tf.nn.relu(conv2+self.conv2_bias)

        with tf.name_scope('Max_Pooling_1'):
            max_pooling_1 = tf.nn.max_pool(conv2_output,
                                           ksize = [1, self.pooling_size,self.pooling_size,1],
                                           strides = [1, self.pooling_stride,self.pooling_stride,1],
                                           padding = "SAME",
                                           name='pooling1')

        with tf.name_scope('conv3_layer'):
            with tf.name_scope('convolution'):
                conv3 = tf.nn.conv2d(input = max_pooling_1,
                                     filter = self.conv3_weight,
                                     strides = [1,self.filter_stride,self.filter_stride,1],
                                     padding = "SAME",
                                     use_cudnn_on_gpu=True,
                                     name='conv3')
                conv3_output = tf.nn.relu(conv3+self.conv3_bias)

        with tf.name_scope('conv4_layer'):
            with tf.name_scope('convolution'):
                conv4 = tf.nn.conv2d(input = conv3_output,
                                     filter = self.conv4_weight,
                                     strides = [1,self.filter_stride,self.filter_stride,1],
                                     padding = "SAME",
                                     use_cudnn_on_gpu=True,
                                     name='conv4')
                conv4_output = tf.nn.relu(conv4+self.conv4_bias)

        with tf.name_scope('Max_Pooling_2'):
            max_pooling_2 = tf.nn.max_pool(conv4_output,
                                           ksize = [1, self.pooling_size, self.pooling_size, 1],
                                           strides = [1, self.pooling_stride, self.pooling_stride, 1],
                                           padding = "SAME",
                                           name='pooling2')

        with tf.name_scope('FC1'):
            shape = max_pooling_2.get_shape().as_list() # 扁平化
            reshape = tf.reshape(max_pooling_2, [shape[0], shape[1] * shape[2] * shape[3]])
            
            
            FC1_output = tf.nn.relu(tf.matmul(reshape, self.fc1_weights)+self.fc1_bias)

        """
        注意: 输出层不加relu激活函数的原因是因为激活函数relu其实会选特征，
             这些对特征的“选择”有可能会导致结果的不准确性
             所以在输出层就直接将结果相加，而不能利用激活函数进行过滤
        """
        with tf.name_scope('FC2'):
            output_layer = tf.matmul(FC1_output,self.fc2_weights)+self.fc2_bias
        
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

    # 2. 初始化网络 + 构建网络
    num_hidden = 16  # 设置隐藏层中的神经元个数为128个
    batch_size = 32  # 每次读入图片的多少
    filter_size = 3    # 滑窗大小
    filter_stride = 1  # 滑窗步长
    num_filter = 16   # filter多少
    pooling_size = 2 # max pooling 滑窗大小
    pooling_stride = 2 # max pooling 滑窗步长

    myCNN = CNN(num_hidden, 
                batch_size,imageSize,num_channel,num_labels, 
                filter_size,filter_stride, num_filter,pooling_size,pooling_stride)
    

    # 3. 构建网络
    # FC.graphConstruction()

    # 4. RUN Model
    myCNN.run(samples_datasets=train_sample,labels_datasets=train_label)

    # 5. Test Model
    myCNN.test_accuracy(samples_datasets=test_sample,labels_datasets=test_label)

    print('down!', 'Total Time is: ', time.time()-start)