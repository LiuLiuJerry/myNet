#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
import collections

import txtReader

# this part is used to make use of codes from third parties
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/../")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus")

import ioUtil
Model = collections.namedtuple("Model",
                               "dist2, mask, dist1, mean_dist"
                               )


def get_transform_Loss0(outputpoints, inputpoints, FLAGS):
    #displacement [batch_size, npts, 3]
    #产生的相近的点的displacement也应该相近    
    bs = FLAGS.batch_size
    npts = FLAGS.point_num_out

    square_dist2 = pairwise_l2_norm2_batch(inputpoints, inputpoints)
    dist2 = tf.sqrt(square_dist2) #[batch_size, npts, npts]
    knndis = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    threds = tf.reduce_min(knndis.values, -1, keep_dims=True) #[batch_size,npts,1]
    threds = tf.abs(tf.tile(threds, [1,1,npts]))  #[batch_size,npts,npts]
    mask = tf.subtract(dist2, threds) <= 0 #是否属于最近的nnk个数 [batch_size, npts, npts]


    square_dist = pairwise_l2_norm2_batch(outputpoints, outputpoints)
    dist1 = tf.sqrt(square_dist)

    dist_inthred = tf.subtract(dist1, threds)
    dist_inthred = tf.maximum(dist_inthred,0)

    mask_threds = tf.boolean_mask(dist_inthred, mask)
    mean_dist = tf.reduce_mean(mask_threds) # 平方和，特征维求平方和，降维    
    
    return Model(dist2 = dist2, threds=threds, mask=mask, dist1=dist1, dist_inthred=dist_inthred, mean_dist=mean_dist)


def get_Geometric_Loss(predictedPts, targetpoints, tags, FLAGS):

    gen_points = FLAGS.generate_num
    #targetpoints = gather_point(targetpoints, farthest_point_sample(gen_points, targetpoints)) #将targetpoints按照输出的点的个数采样
    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt( square_dist ) # 开方
    #在相同tag范围内查找最近点
    tag1 = tf.tile(tf.expand_dims(tags, -1), [1,gen_points])
    tag2 = tf.tile(tf.expand_dims(tags, 0), [gen_points, 1])
    mask = tf.cast(tf.equal(tag1, tag2), dtype=tf.float32)
    mask2 = tf.cast(~tf.equal(tag1, tag2), dtype=tf.float32)
    dist_seg = dist*mask + dist*mask2*100000
    minRow = tf.reduce_min(dist_seg, axis=2) ## 在降维后的第二维，即y那维找最小
    minCol = tf.reduce_min(dist_seg, axis=1) ## 在x那一维找最小值
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol) ## 在[batchsize,x]取平均

    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = tf.sqrt(square_dist2)
    knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk) #返回每行最大的8个数
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    return data_loss, dist_seg, minRow, minCol, shapeLoss

def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1] #point number of each shape
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        ### stack:矩阵拼接  ### tile:某维度重复，如下列代码表示将最后一维重复nump_y次
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y])) 

        yy = tf.expand_dims(y, -1) # [batch_size, npts, 3]
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        
        yy = tf.transpose(yy, perm=[0, 3, 2, 1]) # 交换张量的不同维度相当于1和3维的转置

        diff = tf.subtract(xx, yy) # 做差，xx中每个点和yy中每个点的差
        square_diff = tf.square(diff) # 平方

        square_dist = tf.reduce_sum(square_diff, 2) # 平方和，特征维求平方和，降维

        return square_dist  


# 这个函数把两个参数对应的数据集放进来
Test_examples  = ioUtil.load_examples('../data_hdf5/horse_seg_test.hdf5', 'names')
#segmentation tags
Tags = txtReader.read_txt("../data_hdf5/seg.txt")
tags = np.squeeze(Tags)

shape_in = Test_examples.pointSet_out

inputpoints = shape_in[0:1, ...]
outputpoints = shape_in[7:8, ...]
#inputpoints = np.random.rand(1, 10, 3)
#transform = np.random.rand(1, 10, 3)
#outputpoints = inputpoints + transform

# FLAGS
FLAGS = collections.namedtuple("FLAGS", "gpu, batch_size, point_num_out, point_num_in, range_max, radiusScal")
FLAGS.gpu = 0
FLAGS.batch_size = 1
FLAGS.point_num_out = 10
FLAGS.point_num_in = 19
FLAGS.range_max = 1
FLAGS.radiusScal = 1
FLAGS.nnk = 3
FLAGS.generate_num = 2048
FLAGS.densityWeight = 1

with tf.Graph().as_default():
    sess = tf.Session()

    data_loss, dist_seg, minRow, minCol, shapeLoss = get_Geometric_Loss(inputpoints, outputpoints, tags, FLAGS)
    #t2 = sess.run(test2)
    '''print('input', inputpoints)
    print('output', outputpoints)
    print('dist2', t2.dist2)
    print('dist1', t2.dist1)
    print('mask', t2.mask)
    print('mean_dist',t2.mean_dist)'''
    print(sess.run(dist_seg))
    print(sess.run(minRow))
    print(sess.run(minCol))
    print(sess.run(shapeLoss))

