#!/usr/bin/env python
#coding:utf-8
"""
  Author:  weiyiliu --<weiyiliu@us.ibm.com>
  Purpose: 存储一些深度学习中常用的函数
  Created: 04/02/17
"""
#----------------------------------------------------------------------
def get_chunk_iterator(sampleData, labelDatasets, chunk_size=100):
    """
    读数据的时候按照一块一块的读取
    该函数为一个迭代器，每次yield: i, samples_chunk, label_chunk
        i             : 当前次数
        samples_chunk : 样本块
        label_chunk   : 样本块对应的标签块
    
    @samples    : 输入的 样本
    @labels     : 输入的 样本对应的标签
    @chunk_size : 每次读入的数据块大小, 默认为100
    """
    if len(sampleData) != len(labelDatasets):
        raise Exception("Sample Length != Label Length, Please Check Original Datasets")

    stepStart = 0
    i = 0

    while stepStart < len(sampleData):
        stepEnd = stepStart + chunk_size
        if stepEnd < len(sampleData):
            yield i, sampleData[stepStart:stepEnd], labelDatasets[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd

#----------------------------------------------------------------------
def variable_sumaries(var):
    """
    用于存储变量，放入TensorBoard中
    来源: https://www.tensorflow.org/get_started/summaries_and_tensorboard
    @var : 输入的Tensor
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)