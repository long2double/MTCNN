# coding: utf-8
import tensorflow as tf
import numpy as np
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Detector:
    '''识别多组图片'''
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, [None, data_size, data_size, 3])
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session()
            # 重载模型
            saver = tf.train.Saver()
            # model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(self.sess, model_path)
        self.data_size = data_size
        self.batch_size = batch_size

    def predict(self, databatch):
        batch_size = self.batch_size  # 16
        minibatch = []
        cur = 0
        # 所有数据总数
        n = databatch.shape[0]  # (326, 24, 24, 3)  (1, 48, 48, 3)

        # 将数据整理成固定batch
        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size

        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []

        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            # 最后一组数据不够一个batch的处理,m size as a batch
            if m < batch_size:
                keep_inds = np.arange(m)  # m = 5 keep_inds = [0,1,2,3,4]
                gap = self.batch_size - m  # batch_size = 7, gap = 2
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)  # -3
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            cls_prob, bbox_pred, landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred, self.landmark_pred], feed_dict={self.image_op: data})
            
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])

        """
        2667it [35:42,  1.01it/s]0 0 0
        Traceback (most recent call last):
          File "gen_hard_example.py", line 195, in <module>
            main(parse_arguments(sys.argv[1:]))
          File "gen_hard_example.py", line 72, in main
            detectors,_=mtcnn_detector.detect_face(test_data)
          File "../detection/MtcnnDetector.py", line 90, in detect_face
            boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
          File "../detection/MtcnnDetector.py", line 183, in detect_rnet
            cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
          File "../detection/detector.py", line 65, in predict
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
          File "<__array_function__ internals>", line 6, in concatenate
        ValueError: need at least one array to concatenate
        """
        if len(cls_prob_list) == 0 or len(bbox_pred_list) == 0 or len(landmark_pred_list) == 0:
            return None, None, None
        else:
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)



