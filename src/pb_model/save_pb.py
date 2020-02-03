import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from network.model import o_net 


root_path = os.path.dirname(__file__).split('MTCNN')[0]
project = os.path.dirname(__file__).split('MTCNN')[1]

def save_pb():
    with tf.Session() as sess:
        input_image = tf.placeholder(tf.float32, shape=[None, 48, 48, 3], name='input_image')
        cls_loss, bbox_loss, landmark_loss = o_net(input_image, training=False)
        saver = tf.train.Saver()
        saver.restore(sess, root_path + 'MTCNN' + '/checkpoint/onet/onet-30')
        # ops = tf.get_default_graph().get_operations()
        # print(ops)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['ONet/landmark_fc/BiasAdd'])
        with tf.gfile.FastGFile(os.path.dirname(__file__) + '/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

if __name__ == '__main__':
    save_pb()