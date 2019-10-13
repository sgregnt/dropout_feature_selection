import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys


# plotting stuff
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import copy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# This should match the size of the drop out layer (i.e. the numbr of neurons)
# in the maxpool layer of the main network
# (not the rotation network)
FEATURE_SIZE = 1024

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            mask_pl = MODEL.placeholder_mask(BATCH_SIZE, FEATURE_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, mask=mask_pl, FEATURE_SIZE= FEATURE_SIZE)
            loss, mat_diff = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)



        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        restore = False
        # restore = False#True


        if restore:
            print(LOG_DIR)
            saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt"))
        else:
            # pass
            # Init variables

            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            #sess.run(init)
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'mask_pl': mask_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'mat_diff': mat_diff,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        mask = np.ones(shape=(BATCH_SIZE, FEATURE_SIZE))

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()


            train_one_epoch(sess, ops, train_writer, mask)
            eval_one_epoch(sess, ops, test_writer, mask)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, mask):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)


    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        mat_diff_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['mask_pl']: mask,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val , mat_diff = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred'], ops['mat_diff']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
            mat_diff_sum += mat_diff
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))
        log_string('mat_diff: %f' % (mat_diff_sum / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer, mask):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['mask_pl']: mask,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    return (loss_sum / float(total_seen), (total_correct / float(total_seen)), (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
        
def eval_one_epoch_with_subsample(sess, ops, test_writer, mask, j):
    """ ops: dict mapping from string to tf ops """
    print("start eval_one_epoch_with_subsample")
    is_training = False
    total_correct = 0
    ss_total_correct = 0
    total_seen = 0
    ss_total_seen = 0
    loss_sum = 0
    ss_loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    ss_total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    ss_total_correct_class = [0 for _ in range(NUM_CLASSES)]

    def subsample_point_cloud(a, q, k, batch_size):
        
        indices = []
        
        subsampled_data = np.zeros(shape=a.shape)

        for i in range(k):  # presumably for each i it selects the i-th important point, up to first k-points.
            subsampled_data[range(batch_size), [i] * batch_size, :] = a[range(batch_size), q[:, 0, i], :]
            indices.append(q[0, 0, i])
            # see  https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html for vectorized operations:

        # fill the remaining points with multiple of the last sample
        subsampled_data[range(batch_size), k:a.shape[1], :] = a[range(batch_size), q[:, 0, k - 1], :].reshape(
            (batch_size, 1, 3))
        return subsampled_data, indices

    for fn in range(len(TEST_FILES)):
        print("file", TEST_FILES[fn])
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        
        for batch_idx in range(num_batches):
            print("batch_idx", batch_idx)

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['mask_pl']: mask,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            
            summary, step, loss_val, pred_val, max_layer , b4_max_layer = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred'], ops['max_layer'], ops['b4_max_layer']], feed_dict=feed_dict)


            # evaluate on original data
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

            #-------------------------------------------------------
            # do subsampling and test the results on subsampled data
            #-------------------------------------------------------

            q = np.argmax(b4_max_layer, axis=1)
            # subsample data
            a = current_data[start_idx:end_idx, :, :]
            print("subsample part")
            subsampled_data, indices = subsample_point_cloud(a, q, k=j, batch_size=BATCH_SIZE)
            print("len(np.unique(indices))", len(np.unique(indices)))
            
            # print("subsampled_data.shape", subsampled_data.shape)
            # print("mask.shape", mask.shape)
            # print("subsampled_data[start_idx:end_idx, :, :].shape", subsampled_data.shape)
            # print("current_label[start_idx:end_idx].shape", current_label[start_idx:end_idx].shape)
            # print("start_idx:end_idx", start_idx, end_idx)

            feed_dict = {ops['pointclouds_pl']: subsampled_data,
                         ops['mask_pl']: mask,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}

            summary, step, loss_val, ss_pred_val, ss_max_layer, ss_b4_max_layer = sess.run([ops['merged'], ops['step'],
                                                                                   ops['loss'], ops['pred'],
                                                                                   ops['max_layer'],
                                                                                   ops['b4_max_layer']],
                                                                                  feed_dict=feed_dict)
            # evaluate on original data
            ss_pred_val = np.argmax(ss_pred_val, 1)
            correct = np.sum(ss_pred_val == current_label[start_idx:end_idx])
            ss_total_correct += correct
            ss_total_seen += BATCH_SIZE
            ss_loss_sum += (loss_val * BATCH_SIZE)
            print("loss_sum", loss_sum, "ss_loss_sum", ss_loss_sum)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                ss_total_seen_class[l] += 1
                ss_total_correct_class[l] += (ss_pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    part = ((loss_sum / float(total_seen), (total_correct / float(total_seen)),np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    log_string('eval mean loss: %f' % (ss_loss_sum/ float(ss_total_seen)))
    log_string('eval accuracy: %f' % (ss_total_correct / float(ss_total_seen)))

    log_string('eval avg class acc: %f' % (np.mean(np.array(ss_total_correct_class)/np.array(ss_total_seen_class,dtype=np.float))))
    ss_part = ((ss_loss_sum / float(ss_total_seen), (ss_total_correct / float(ss_total_seen)), np.mean(np.array(ss_total_correct_class) / np.array(ss_total_seen_class, dtype=np.float))))
    return part, ss_part

def evaluate_mask():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            mask_pl = MODEL.placeholder_mask(BATCH_SIZE, FEATURE_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, mask=mask_pl)
            loss, mat_diff = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        restore = True

        if restore:
            print(LOG_DIR)
            saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt"))
            print("Restored")
        else:
            # pass
            # Init variables

            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            # sess.run(init)
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'mask_pl': mask_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'b4_max_layer' : end_points['b4_max_layer'],
               'max_layer' : end_points['max_layer'],
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        mean_loss_res_top = []
        ss_mean_loss_res_top = []
        mean_loss_res_b = []
        ss_mean_loss_res_b = []
        acc_res_top = []
        ss_acc_res_top = []
        acc_res_b = []
        ss_acc_res_b = []
        class_acc_res_top = []
        ss_class_acc_res_top = []
        class_acc_res_b = []
        ss_class_acc_res_b = []
        k = 20
        stretch = 200
        plot_range = list(range(0, stretch, k))
        plot_range = [100, 200, 300, 400, 600, 800, 1024]
        # for i in range(0, stretch, k):
        for j in [100, 200, 300, 400, 600, 800, 1024]:
            print(j)
            log_string('**** feature %03d removed ****' % (j))
            sys.stdout.flush()
            # mask = np.zeros(shape=(BATCH_SIZE, FEATURE_SIZE))
            # if i > 0:
            #     mask[:, -(i+k):-i] = 1
            # mean_loss, acc, class_acc = eval_one_epoch(sess, ops, test_writer, mask)
            # mean_loss_res_b.append(mean_loss)
            # acc_res_b.append(acc)
            # class_acc_res_b.append(class_acc)
            mask = np.ones(shape=(BATCH_SIZE, FEATURE_SIZE))
            # if i > 0:
            # mask[:, i:(i+k)] = 0
            print("range", j, j+k)
            # mean_loss, acc, class_acc = eval_one_epoch(sess, ops, test_writer, mask)
            
            part, ss_part = eval_one_epoch_with_subsample(sess, ops, test_writer, mask, j)

            mean_loss, acc, class_acc =part
            mean_loss_res_top.append(mean_loss)
            acc_res_top.append(acc)
            class_acc_res_top.append(class_acc)

            ss_mean_loss, ss_acc, ss_class_acc = ss_part
            ss_mean_loss_res_top.append(ss_mean_loss)
            ss_acc_res_top.append(ss_acc)
            ss_class_acc_res_top.append(ss_class_acc)

    plt.figure()
    plt.plot(plot_range, mean_loss_res_top, c= 'r')
    plt.plot(plot_range, ss_mean_loss_res_top, c= 'g')
    # plt.plot(plot_range, mean_loss_res_b, c = 'b')
    # plt.plot(plot_range, ss_mean_loss_res_b, c = 'y')
    plt.xlabel('neuron')
    plt.ylabel('mean loss')
    plt.title('removing neurons from opposite ends')

    plt.figure()
    plt.plot(plot_range, acc_res_top, c='r')
    plt.plot(plot_range, ss_acc_res_top, c='g')
    # plt.plot(plot_range, acc_res_b, c='b')
    # plt.plot(plot_range, ss_acc_res_b, c='y')
    plt.xlabel('neuron')
    plt.ylabel('accuracy')
    plt.title('removing neurons from opposite ends')
    plt.figure()
    plt.plot(plot_range, class_acc_res_top, c='r')
    plt.plot(plot_range, ss_class_acc_res_top, c='g')
    # plt.plot(plot_range, class_acc_res_b, c='b')
    # plt.plot(plot_range, ss_class_acc_res_b, c='y')
    plt.xlabel('neuron')
    plt.ylabel('class accuracy')
    plt.title('removing neurons from opposite ends')
    plt.show()

if __name__ == "__main__":
    train()
    #evaluate_mask()
    LOG_FOUT.close()
