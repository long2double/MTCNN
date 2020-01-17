# -*-coding:utf-8-*-
from utils import *
import sys
sys.path.append('../')
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.python.platform import gfile
from time import time


def inference():
    sess = tf.Session()
    with gfile.FastGFile('model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # ops = tf.get_default_graph().get_operations()
        # print(ops)

    sess.run(tf.global_variables_initializer())
    input_image = sess.graph.get_tensor_by_name('input_image:0')
    landmark = sess.graph.get_tensor_by_name('ONet/landmark_fc/BiasAdd:0')

    data_file = "/mnt/data/changshuang/data/flickr/"
    anno_file = "/mnt/data/changshuang/data/aflw_anno.txt"
    # data: {'images': images, 'bboxes': bboxes, 'landmarks':landmarks}
    data = read_annotation(data_file, anno_file)
    img_data = list(zip(data["images"], data["bboxes"], data["landmarks"]))
    for img_path, img_bbox, img_landmarks in img_data:
        img = cv2.imread(img_path)
        bbox = np.array(img_bbox)

        dets = convert_to_square(bbox)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        h, w, c = img.shape
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):  # 17
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        t1 = time()
        # batch_size = 16
        # minibatch = []
        # cur = 0
        # n = cropped_ims.shape[0]
        # while cur < n:
        #     minibatch.append(cropped_ims[cur: min(cur + batch_size, n), :, :, :])
        #     cur += batch_size
        #
        # landmark_pred_list = []
        # for data in minibatch:
        #     m = data.shape[0]
        #     real_size = batch_size
        #     # 最后一组数据不够一个batch的处理,m size as a batch
        #     if m < batch_size:
        #         keep_inds = np.arange(m)  # m = 5 keep_inds = [0,1,2,3,4]
        #         gap = batch_size - m  # batch_size = 7, gap = 2
        #         while gap >= len(keep_inds):
        #             gap -= len(keep_inds)  # -3
        #             keep_inds = np.concatenate((keep_inds, keep_inds))
        #         if gap != 0:
        #             keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
        #         data = data[keep_inds]
        #         real_size = m
        #     pre_landmarks = sess.run(landmark, feed_dict={input_image: data})
        #     landmark_pred_list.append(pre_landmarks[:real_size])
        # if len(landmark_pred_list) == 0:
        #     continue
        # else:
        #     pre_landmarks = np.concatenate(landmark_pred_list, axis=0)
        pre_landmarks = sess.run(landmark, feed_dict={input_image: cropped_ims})
        print(time() - t1)

        w = bbox[:, 2] - bbox[:, 0] + 1
        h = bbox[:, 3] - bbox[:, 1] + 1
        pre_landmarks[:, 0::2] = (np.tile(w, (5, 1)) * pre_landmarks[:, 0::2].T + np.tile(bbox[:, 0], (5, 1)) - 1).T
        pre_landmarks[:, 1::2] = (np.tile(h, (5, 1)) * pre_landmarks[:, 1::2].T + np.tile(bbox[:, 1], (5, 1)) - 1).T

        for i in range(bbox.shape[0]):
            box_gt = bbox[i, :4]
            corpbbox_gt = [int(box_gt[0]), int(box_gt[1]), int(box_gt[2]), int(box_gt[3])]
            # 画人脸框
            cv2.rectangle(img, (corpbbox_gt[0], corpbbox_gt[1]), (corpbbox_gt[2], corpbbox_gt[3]), (0, 225, 255), 2)
        # 画关键点
        for i in range(pre_landmarks.shape[0]):
            for j in range(len(pre_landmarks[i]) // 2):
                cv2.circle(img, (int(pre_landmarks[i][2 * j]), int(int(pre_landmarks[i][2 * j + 1]))), 3, (0, 0, 255), -1)
                cv2.circle(img, (int(img_landmarks[i][2 * j]), int(int(img_landmarks[i][2 * j + 1]))), 3, (0, 255, 255), -1)
        cv2.imshow('show image', img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()

