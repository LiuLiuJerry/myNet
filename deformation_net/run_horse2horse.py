#coding:utf-8
import numpy as np
import sys
import os
import argparse
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/../")
sys.path.append(   BASE_DIR + "/../pointnet_plusplus")
import horse2horse
import ioUtil
import logging
import os
from tensorflow.python.client import device_lib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print(tf.test.is_gpu_available())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print(device_lib.list_local_devices())


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument('--train_hdf5', default='../data_hdf5/horse_train.hdf5')
parser.add_argument('--test_hdf5', default='../data_hdf5/horse_test.hdf5')

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use [default: 0]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--epoch', type=int, default=1000, help='number of epoches to run [default: 200]')
parser.add_argument('--decayEpoch',  type=int, default=50, help='steps(how many epoches) for decaying learning rate')

parser.add_argument("--range_max", type=float, default=1.0, help="max length of point displacement[default: 1.0]")
parser.add_argument("--radiusScal", type=float, default=1.0, help="a constant for scaling radii in pointnet++ [default: 1.0]")	       
parser.add_argument('--checkpoint', default=None, help='epoch_##.ckpt')
parser.add_argument("--densityWeight", type=float, default=1.0, help="density weight [default: 1.0]")
parser.add_argument("--nnk", type=int, default=8, help="density:  number of nearest neighbours [default: 8]")

###  None  None  None
parser.add_argument('--point_num_in', type=int, default=None, help='do not set the argument')
parser.add_argument('--point_num_out', type=int, default=None, help='do not set the argument')
parser.add_argument('--generate_num', type=int, default=2048, help='do not set the argument')
parser.add_argument('--example_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--output_dir', type=str,  default=None, help='do not set the argument')

# for learning
parser.add_argument('--LEARNING_RATE_G', type=float, default=0.001, help='could be changed')
parser.add_argument('--ADAM_BETA_D', type=float, default=0.5, help='could be changed')
parser.add_argument('--ADAM_BETA_G', type=float, default=0.5, help='could be changed')

FLAGS = parser.parse_args()

# 这个函数把两个参数对应的数据集放进来
Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, 'names')
Test_examples  = ioUtil.load_examples(FLAGS.test_hdf5, 'names')


'''重新排列数据'''
Train_data = ioUtil.arrange_datas(Train_examples)
Test_data = ioUtil.arrange_datas(Test_examples)

############# FALG things #################################
FLAGS.point_num_in = Train_examples.skeleton_in.shape[1] # shape gives nums in every dims
FLAGS.point_num_out = Train_examples.pointSet_out.shape[1]

FLAGS.example_num = Train_examples.skeleton_in.shape[0]
EXAMPLE_NUM = FLAGS.example_num

TRAINING_EPOCHES = FLAGS.epoch

batch_size = FLAGS.batch_size


##################### output data #######################
output_dir = 'output_horse'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
     os.mkdir(MODEL_STORAGE_PATH)
 
SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
     os.mkdir(SUMMARIES_FOLDER)

ioUtil.output_point_cloud_ply( Test_examples.skeleton_in, Test_examples.names, output_dir, 'gt_'+ 'skeleton')
ioUtil.output_point_cloud_ply( Test_examples.pointSet_out, Test_examples.names, output_dir, 'gt_'+ 'surface')
# print arguments
for k, v in FLAGS._get_kwargs():
    print(k + ' = ' + str(v) )

# define train function
def train():
    with tf.Graph().as_default():
        with tf.device('/device:GPU:0'):
            model = horse2horse.create_model(FLAGS)
        
        ########### init ans configure #########################
        saver = tf.train.Saver( max_to_keep=5)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        sess = tf.Session(config=config)
        ### run了 所有global Variable 的 assign op
        init = tf.global_variables_initializer()
        sess.run(init)

        # restore variables from disk
        start_epoch_number = 1
        if FLAGS.checkpoint is not None: # check point: 二进制文件，它包含的权重变量，biases变量和其他变量
            print('load checkpoint: ' + FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint )

            fname = os.path.basename( FLAGS.checkpoint)
            start_epoch_number = int( fname[6:-5] ) + 1# 从之前训练的地方接着开始训练。[ : ]是字符串截取的意思

            print( 'Start_epoch_number - ' + str(start_epoch_number) )

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer= tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(output_dir, 'arguments.txt'), 'w') # write arguments done
        fcmd.write(str(FLAGS))
        #pickle.dump(fcmd, FLAGS)
        fcmd.close()


        ############ train one epoch #############################################################
        def train_one_epoch(epoch_num):
            is_training = True

            Train_examples_shuffled = ioUtil.shuffle_datas(Train_data)

            skeleton_in = Train_examples_shuffled.skeleton_in
            pointSet_in = Train_examples_shuffled.pointSet_in
            pointSet_out = Train_examples_shuffled.pointSet_out

            #print('skeleton_in', skeleton_in.shape)
            #print('pointSet_in', pointSet_in.shape)
  
            num_data = skeleton_in.shape[0]
            num_batch = num_data // batch_size  #双斜杠（//）表示地板除，即先做除法（/），然后向下取整（floor）

            # losses 
            gen_loss = 0.0

            # for every batch
            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                # input data
                feed_dict = {
                    model.skeleton_in_ph: skeleton_in[begidx: endidx, ...],
                    model.pointSet_in_ph: pointSet_in[begidx: endidx, ...],
                    model.pointSet_out_ph: pointSet_out[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }
                # loss and other variables computed by sess
                fetches = {
                    "gen_loss": model.gen_loss,
                    "gen_shapeLoss": model.gen_shapeLoss,
                    "gen_densityLoss": model.gen_densityLoss,
                    "learning_rate_G": model.learning_rate_G,
                    "train_gen": model.train_gen,
                    "global_step" : model.global_step, # add 1 per time 
                }
                ### session是抽象模型的实现者 
                results = sess.run(fetches, feed_dict = feed_dict)
                # get loss 
                gen_loss += results["gen_loss"]

                # print every 50 batches 
                if j % 50 == 0:
                    print('    ' + str(j) + '/' + str(num_batch) + ':    '  )
                    print('            gen_loss = {:.4f},'.format(results["gen_loss"] ) )
                    print('            gen_shapeLoss = {:.4f},'.format(results["gen_shapeLoss"] ) )
                    print('            gen_densityLoss = {:.4f},'.format(results["gen_densityLoss"] ) )
                    print('            learning_rate_G = {:.6f}'.format(results["learning_rate_G"] )  )
                    print('            global_step = {0}'.format(results["global_step"] )  )
            # evaluate summaries # 每当所有数组都训练完一遍时，来一遍总结
            gen_loss /= num_batch
            # summary the model every epoch
            training_sum = sess.run(model.training_sum_ops,
                                    feed_dict={ 
                                                model.train_gen_loss_ph:gen_loss
                                                } )
            train_writer.add_summary(training_sum, epoch_num)
            print('\n\t<< summary: gen_loss     = %.4f\n' % gen_loss)
        ################## end of training function ##################################################

        ###################### test function #########################################################
        def eval_one_epoch(epoch_num, mustSavePly = False):
            is_training = False


            skeleton_in = Test_data.skeleton_in
            pointSet_in = Test_data.pointSet_in
            pointSet_out = Test_data.pointSet_out
            names       = Test_data.names

            num_data = skeleton_in.shape[0]
            num_batch = num_data // batch_size

            # losses 
            gen_loss = 0.0

            # for every batch
            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                # input data
                feed_dict = {
                    model.skeleton_in_ph: skeleton_in[begidx: endidx, ...],
                    model.pointSet_in_ph: pointSet_in[begidx: endidx, ...],
                    model.pointSet_out_ph: pointSet_out[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }
                # loss and other variables computed by sess

                fetches = {
                    "gen_loss": model.gen_loss,
                    "predictedSet": model.predictedSet
                }
                results = sess.run(fetches, feed_dict = feed_dict)
                # get loss 
                gen_loss     += results["gen_loss"]

                # write test results
                if epoch_num  % 20 == 0  or  mustSavePly:
                    # 多算计次然后取个平均
                    # save predicted point sets with 1 single feeding pass
                    nametosave = names[begidx: endidx, ...]
                    Predicted_xyz = np.squeeze(np.array(results["predictedSet"]))

                    ioUtil.output_point_cloud_ply(Predicted_xyz, nametosave, output_dir,
                                                    'Ep' + str(epoch_num) + '_predicted_' + 'X1')

                    # save predicted point sets with 4 feeding passes
                    for i in range(3):
                        results = sess.run(fetches, feed_dict=feed_dict)
                        Predicted_xyz__ = np.squeeze(np.array(results["predictedSet"]))
                        Predicted_xyz = np.concatenate((Predicted_xyz, Predicted_xyz__), axis=1)

                    ioUtil.output_point_cloud_ply(Predicted_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + 'X4')

                    # save predicted point sets with 8 feeding passes
                    for i in range(4):
                        results = sess.run(fetches, feed_dict=feed_dict)
                        Predicted_xyz__ = np.squeeze(np.array(results["predictedSet"]))
                        Predicted_xyz = np.concatenate((Predicted_xyz, Predicted_xyz__), axis=1)

                    ioUtil.output_point_cloud_ply( Predicted_xyz, nametosave, output_dir,
                                                   'Ep' + str(epoch_num) + '_predicted_' + 'X8')
            
            # summary the model every epoch
            # evaluate summaries # 每当所有数组都训练完一遍时，来一遍总结
            gen_loss /= num_batch

            testing_sum = sess.run(model.testing_sum_ops,
                                    feed_dict={ model.test_gen_loss_ph:gen_loss
                                                } )

            test_writer.add_summary(testing_sum, epoch_num)

            # inform
            print('\n\t<< summary: gen_loss     = %.4f\n' % gen_loss)

        ################################# end if test function ################################################################
        # use the function defined before
        if FLAGS.mode == 'train':
            for epoch in range(start_epoch_number, TRAINING_EPOCHES+1):
                print(' \n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
                train_one_epoch(epoch)

                if epoch % 20 == 0:
                    cp_Filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
                    print( 'Successfilly store the checkpoint model into \n' + cp_Filename)

                    print( '<<< Testing on the test dataset...\n')
                    eval_one_epoch(epoch, mustSavePly=True)
        else:
            print( '<<< Testing on the test dataset ...')
            eval_one_epoch(start_epoch_number, mustSavePly=True)


# main function
if __name__ == '__main__':
    train()



