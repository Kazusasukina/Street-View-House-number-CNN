#!/usr/bin/env python
#coding:utf-8
"""
  Author:  weiyiliu --<weiyiliu@us.ibm.com>
  Purpose: 装载数据 --- 来源于网络
  Created: 04/02/17
"""

from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

def reformat(samples, labels):
    # 改变原始数据的形状
    #  0       1       2      3          3       0       1      2
    # (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

    # labels 变成 one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 , represented as 10
    # labels 变成 one-hot encoding, [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])	# slow code, whatever
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels

def normalize(samples):
    '''
    并且灰度化: 从三色通道 -> 单色通道     省内存 + 加快训练速度
    (R + G + B) / 3
    将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
    @samples: numpy array
    '''
    a = np.add.reduce(samples, keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)
    a = a/3.0
    return a/128.0 - 1.0



# --------------------
# Written By Myself
# 2017-04-02
#----------------------------------------------------------------------
def SetImageProperty(automaticSet = True):
    """
    设置图片的imageSize, num_channels, labels属性
    """
    if automaticSet is True:
        imageSize = 32
        num_channel = 1
        num_labels = 10
    else:
        imageSize = int(input('请输入正方形图片的尺寸:\t'))
        num_channel = int(input('请输入该图片的通道个数:\t'))
        num_labels = int(input('请输入该图片可能所属的标签个数:\t'))

    return imageSize,num_channel,num_labels

#----------------------------------------------------------------------
def loadDatasets(train_path,test_path):
    """
    读入datasets所在的PATH
    @train_path 训练集  Path
    @test_path  测试集  Path
    """
    train_data = load(train_path)
    test_data  = load(test_path)
    
    #refomat samples 改变原始数据的形状
    format_train_sample, format_train_label = reformat(train_data['X'],train_data['y'])
    format_test_samples, format_test_labels = reformat(test_data['X'], test_data['y'])
    
    # normalize 将图片灰度化
    nor_format_train_sample = normalize(format_train_sample)
    nor_format_test_sample  = normalize(format_test_samples)
    
    return nor_format_train_sample, format_train_label, nor_format_test_sample, format_test_labels