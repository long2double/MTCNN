import tensorflow as tf
from tensorflow.contrib import slim
from network.loss import *


def p_net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''pnet的结构'''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d], activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            net = slim.conv2d(inputs, 10, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 16, 3, scope='conv2')
            net = slim.conv2d(net, 32, 3, scope='conv3')
            # 二分类输出通道数为2
            conv4_1 = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax, scope='conv4_1')  # [batch, 1, 1, 2]
            bbox_pred = slim.conv2d(net, 4, 1, activation_fn=None, scope='conv4_2')  # [batch, 1, 1, 4]
            landmark_pred = slim.conv2d(net, 10, 1, activation_fn=None, scope='conv4_3')  # [batch, 1, 1 10]
            
            if training:
                cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')  # [batch, 2]
                cls_loss = cls_ohem(cls_prob, label)
                
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')  # [bacth,4]
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                
                landmark_pred = tf.squeeze(landmark_pred, [1, 2], name='landmark_pred')  # [batch,10]
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                
                accuracy = cal_accuracy(cls_prob, label)
                L2_loss = tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            else:
                # 测试时batch_size=1
                cls_pro_test = tf.squeeze(conv4_1, axis=0)  # [H, W, 2]
                bbox_pred_test = tf.squeeze(bbox_pred, axis=0)  # [H, W, 4]
                landmark_pred_test = tf.squeeze(landmark_pred, axis=0)   # [H, W, 10]
                return cls_pro_test, bbox_pred_test, landmark_pred_test


def r_net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''RNet结构'''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d], activation_fn=prelu, weights_initializer=slim.xavier_initializer(), weights_regularizer=slim.l2_regularizer(0.0005), padding='VALID'):
            net = slim.conv2d(inputs, 28, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 48, 3, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 2, scope='conv3')
            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope='fc1')
            cls_prob = slim.fully_connected(fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')  # [batch, 2]
            bbox_pred = slim.fully_connected(fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')  # [batch, 4]
            landmark_pred = slim.fully_connected(fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')  # [batch, 10]
            if training:
                cls_loss = cls_ohem(cls_prob, label)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                accuracy = cal_accuracy(cls_prob, label)
                L2_loss = tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            else:
                return cls_prob, bbox_pred, landmark_pred


def o_net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''ONet结构'''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d], activation_fn=prelu, weights_initializer=slim.xavier_initializer(), weights_regularizer=slim.l2_regularizer(0.0005), padding='VALID'):
            net = slim.conv2d(inputs, 32, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[3,3], stride=2, padding='SAME',scope='pool1')
            net = slim.conv2d(net, 64, 3, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 3, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool3')
            net = slim.conv2d(net, 128, 2, scope='conv4')
            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope='fc1')
            cls_prob = slim.fully_connected(fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')
            bbox_pred = slim.fully_connected(fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')
            landmark_pred = slim.fully_connected(fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')
            if training:
                cls_loss = cls_ohem(cls_prob, label)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                accuracy = cal_accuracy(cls_prob, label)
                L2_loss = tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            else:
                return cls_prob, bbox_pred, landmark_pred


