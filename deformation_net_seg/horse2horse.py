#coding:utf-8
import os
import sys
import collections

# this part is used to make use of codes from third parties
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/../pointnet_plusplus/utils")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus/tf_ops")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus/tf_ops/grouping")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus/tf_ops/sampling")

import tensorflow as tf
import numpy as np # Nummeric python
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_deconv
from tf_sampling import farthest_point_sample, gather_point
from transform_nets import input_transform_net

tf.reset_default_graph()

# a list of names, use the name to visit some corresponding data
Model = collections.namedtuple("Model",
                               "skeleton_in_ph, pointSet_in_ph, pointSet_out_ph, \
                               is_training_ph,\
                               predictedSet, \
                               gen_loss, gen_shapeLoss, gen_densityLoss\
                               train_gen, \
                               learning_rate_G,  global_step,  bn_decay, \
                               training_sum_ops, testing_sum_ops, \
                               train_gen_loss_ph, \
                               test_gen_loss_ph,\
                               tags"
                               )

def create_model( FLAGS ):

    ############################################################
    ####################  Hyper-parameters   ####################
    ##############################################################
    # 损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动加一的操作
    global_step = tf.train.get_or_create_global_step() 
    # 指数衰减学习率 首先使用较大学习率(目的：为快速得到一个比较优的解), 然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定)
 
    learning_rate_G = tf.train.exponential_decay( 
        FLAGS.LEARNING_RATE_G, 
        global_step   * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num  * FLAGS.decayEpoch,  # step size
        0.5,  # decay rate
        staircase=True
    )    
    learning_rate_G = tf.maximum(learning_rate_G, 1e-5)

    bn_momentum = tf.train.exponential_decay(
        0.5,
        global_step  * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num * FLAGS.decayEpoch * 2,     # step size,
        0.5,   # decay rate
        staircase=True
    )
    bn_decay = tf.minimum(0.99,   1 - bn_momentum) # bn : batch normalization


    ##############################################################
    ####################  Create the network  ####################
    ##############################################################
    # placeholder 与 feed_dict={} 是绑定在一起出现的 
    # Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **})
    pointSet_in_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_out, 3) )
    skeleton_in_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_in, 3) )
    pointSet_out_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_out, 3) )
    is_training_ph = tf.placeholder( tf.bool, shape=() )
    tags = tf.placeholder( tf.int32, shape=(FLAGS.point_num_out) )

    ################## define my own netwoek #####################
    ske_features = encoder(skeleton_in_ph, FLAGS, is_training_ph, bn_decay)
    displacements = get_displacements(pointSet_in_ph, ske_features, is_training_ph, FLAGS, bn_decay=None)
    predictedSet = pointSet_in_ph + displacements
    
    ################### define my losses #############################
    gen_geo_loss, gen_shapeLoss, gen_densityLoss = get_Geometric_Loss(predictedSet, pointSet_out_ph, tags, FLAGS)
    transform_Loss = get_transform_Loss(predictedSet, pointSet_in_ph, FLAGS)

    gen_loss = gen_geo_loss + transform_Loss
    #gen_loss = gen_geo_loss
    #gen_loss = gen_shapeLoss + transform_Loss

    # optimize variables
    train_variables = tf.trainable_variables()
    # lambda : 表达式的意思
    
    # trained datas
    train_gen_op = tf.train.AdamOptimizer(learning_rate_G, beta1=FLAGS.ADAM_BETA_G, beta2=0.9).minimize(gen_loss, var_list=train_variables, global_step=global_step)

    train_gen     = train_gen_op
  
    ##############################################################
    ####################  Create summarizers  ####################
    ##############################################################
    ### note:
    # summary的操作都是对某个tensor产生单个的summary protocol buffer，是一种能被tensorboard读取的格式。
    # summary的操作对于整个图来说相当于是外设，因为图的结果并不依赖于summary操作，所以summary操作需要被run
    
    # 告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，不过我正式运行的时候会补上的！
    train_gen_loss_ph = tf.placeholder(tf.float32, shape=())

    test_gen_loss_ph = tf.placeholder(tf.float32, shape=())
    
    # tf.summary.scalar: 用来显示标量信息, 返回值：一个字符串类型的标量张量，包含一个Summaryprotobuf
    lr_G_sum_op = tf.summary.scalar('learning rate_G', learning_rate_G)
    global_step_sum_op = tf.summary.scalar('batch_number', global_step)

    train_gen_loss_sum_op = tf.summary.scalar('train_gen_loss', train_gen_loss_ph) 
    test_gen_loss_sum_op = tf.summary.scalar('test_gen_loss', test_gen_loss_ph) 

    # 把图中所有的summary数据合并在一起，一个run就能启动所有的summary operations

    training_sum_ops = tf.summary.merge( \
        [lr_G_sum_op, \
        train_gen_loss_sum_op])
    test_sum_ops = tf.summary.merge( \
        [test_gen_loss_sum_op])
    # summarize data into Model(tuple) 整个图经常需要检测许许多多的值
    return Model(
                 skeleton_in_ph=skeleton_in_ph,  pointSet_in_ph=pointSet_in_ph,  pointSet_out_ph=pointSet_out_ph,
                 is_training_ph=is_training_ph,
                 predictedSet=predictedSet,
                 gen_loss=gen_loss,   gen_shapeLoss=gen_shapeLoss,    gen_densityLoss=gen_densityLoss,
                 train_gen=train_gen, 
                 learning_rate_G=learning_rate_G, 
                 global_step=global_step,                               bn_decay=bn_decay,
                 training_sum_ops=training_sum_ops,                     testing_sum_ops=test_sum_ops,
                 train_gen_loss_ph     = train_gen_loss_ph,
                 test_gen_loss_ph      = test_gen_loss_ph,
                 tags = tags
    )
    
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
    return data_loss, shapeLoss, densityLoss

def get_Geometric_Loss1(predictedPts, targetpoints, tags, FLAGS):

    gen_points = FLAGS.generate_num
    targetpoints = gather_point(targetpoints, farthest_point_sample(gen_points, targetpoints)) #将targetpoints按照输出的点的个数采样
    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt( square_dist ) # 开方

    #在相同tag范围内查找最近点
    tag1 = tf.tile(tf.expand_dims(tags, -1), [1,gen_points])
    tag2 = tf.tile(tf.expand_dims(tags, 0), [gen_points, 1])
    mask = tf.cast(tf.equal(tag1, tag2), dtype=tf.float32)
    mask2 = tf.cast(~tf.equal(tag1, tag2), dtype=tf.float32)
    dist_seg = dist*mask + dist*100*mask2
    minRow = tf.reduce_min(dist_seg, axis=2) ## 在降维后的第二维，即y那维找最小
    minCol = tf.reduce_min(dist_seg, axis=1) ## 在x那一维找最小值
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol) ## 在[batchsize,x]取平均

    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = tf.sqrt(square_dist2)
    knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk) #返回每行最大的8个数  列是target
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    densityLoss1 = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # density loss on source shape
    square_dist3 = pairwise_l2_norm2_batch(predictedPts, predictedPts)
    dist3 = tf.sqrt( square_dist3+1e-12 ) # 开方
    dist4 = tf.transpose(dist, perm=[0, 2, 1]) #按照predicted找最小
    knndis3 = tf.nn.top_k(tf.negative(dist3), k=FLAGS.nnk) #返回每行最大的8个数
    knndis4 = tf.nn.top_k(tf.negative(dist4), k=FLAGS.nnk)
    densityLoss2 = tf.reduce_mean(tf.abs(knndis3.values - knndis4.values))

    densityLoss = densityLoss1 + densityLoss2

    data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    return data_loss, shapeLoss, densityLoss


def get_transform_Loss(outputpoints, inputpoints, FLAGS):
    #displacement [batch_size, npts, 3]
    #产生的相近的点的displacement也应该相近    
    bs = FLAGS.batch_size
    npts = FLAGS.point_num_out
    R = FLAGS.radiusScal *0.02

    square_dist1 = pairwise_l2_norm2_batch(inputpoints, inputpoints)
    dist1 = tf.sqrt(square_dist1+1e-12) #[batch_size, npts, npts]
    #dist1 = square_dist1
    mask = dist1 < R

    square_dist2 = pairwise_l2_norm2_batch(outputpoints, outputpoints)
    dist2 = tf.sqrt(square_dist2+1e-12)
    #dist2 = square_dist2

    sub_dist = tf.abs(dist2-dist1)
    dist_nei = tf.boolean_mask(sub_dist, mask)

    mean_dist = tf.reduce_mean(dist_nei) # 平方和，特征维求平方和，降维

    return mean_dist

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
    
def encoder(input_points, FLAGS, is_training, bn_decay=None):

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num_in

    l0_xyz = input_points
    l0_points = None
    # generate new shape
    # Set Abstraction layers 
    # abstract from skeleton 
    # original PointNet, don't sample the points l4.points=[batch_size * 1024]
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[32, 32, 64], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='sa_layer4')
    l4_points = tf.squeeze(l4_points)
    return l4_points


def generate(features, FLAGS, is_training, bn_decay=None):

    gen_points = FLAGS.generate_num
    ### 将1×1024的特征转变为点云的坐标 全连接层，生成256个点， 每个点256维特征
    net = tf.reshape(features, [FLAGS.batch_size, 64])
    net = tf_util.fully_connected(net, 1024, scope='G_full_conn1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, gen_points*3, scope='G_full_conn2', bn_decay=bn_decay, activation_fn=None)
    net = tf.reshape(net, [FLAGS.batch_size, gen_points, 3])

    return net



def get_displacements(input_points, ske_features, is_training, FLAGS, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num_out

    point_cloud = input_points

    l0_xyz = point_cloud
    l0_points = None

    # Set Abstraction layers 第一从次2048个点提取1024个点
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')  ### 最后一个变量scope相当于变量前缀
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2* FLAGS.radiusScal, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4* FLAGS.radiusScal, nsample=64,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # PointNet
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

     ### Feature Propagation layers  #################  featrue maps are interpolated according to coordinate  ################     
    # 根据l4的特征值差值出l3
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    # 加入提取的skeleton特征 
    # ske_features : batch_size * featrues
    ske_features = tf.tile(tf.expand_dims(ske_features, 1), [1, num_points, 1])
    l0_points = tf.concat([l0_points, ske_features], axis=-1)
    # 特征转变成 displacement
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements

    
    
   
