# -*-coding:utf-8-*-
import sys
sys.path.append('../')
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train.model import P_Net, R_Net, O_Net
import cv2
import numpy as np
from detection.MtcnnDetector import MtcnnDetector
from time import time


def evaluate(evalu):
    batch_size = 1
    detectors = [None, None, None]
    # model_path = ['../checkpoint/PNet/', '../checkpoint/RNet/', '../checkpoint/ONet']
    model_path = ['../checkpoint/PNet/PNet-30', '../checkpoint/RNet/RNet-22', '../checkpoint/ONet/ONetrefine-30']

    detectors[0] = FcnDetector(P_Net, model_path[0])
    detectors[1] = Detector(R_Net, 24, batch_size, model_path[1])
    detectors[2] = Detector(O_Net, 48, batch_size, model_path[2])

    mtcnn_detector = MtcnnDetector(detectors)

    data_file = "/mnt/data/changshuang/data/flickr/"
    anno_file = "/mnt/data/changshuang/data/aflw_anno.txt"
    # data: {'images': images, 'bboxes': bboxes, 'landmarks':landmarks}
    data = read_annotation(data_file, anno_file)

    img_data = list(zip(data["images"], data["bboxes"], data["landmarks"]))

    local_nme = []
    for img_path, img_bbox, img_landmarks in img_data:
        img = cv2.imread(img_path)
        img_bbox = np.array(img_bbox)
        if evalu == "onet":
            t1 = time()
            boxes_c, landmarks = mtcnn_detector.evaluate_onet(img, img_bbox)  # 1.5ms  time:0.001844(s)
            print("time:%f(s)" % (time() - t1))
        else:
            t1 = time()
            boxes_c, landmarks = mtcnn_detector.detect(img)  # 300ms  time:0.254378(s)
            print("time:%f(s)" % (time() - t1))

        nme = calculate_nme(img_landmarks, landmarks, img_bbox)
        if nme != []:
            local_nme.extend(nme)

        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

            box_gt = img_bbox[i, :4]
            corpbbox_gt = [int(box_gt[0]), int(box_gt[1]), int(box_gt[2]), int(box_gt[3])]
            # 画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0, 0, 255), 2)
            cv2.rectangle(img, (corpbbox_gt[0], corpbbox_gt[1]), (corpbbox_gt[2], corpbbox_gt[3]), (0, 225, 255), 2)
            # 判别为人脸的置信度
            # cv2.putText(img, '{:.2f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 画关键点
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 3, (0, 0, 255), -1)
                cv2.circle(img, (int(img_landmarks[i][2 * j]), int(int(img_landmarks[i][2 * j + 1]))), 3, (0, 255, 255), -1)
        cv2.imshow('show image', img)
        k = cv2.waitKey() & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

    each_landmark = np.array(local_nme).T
    each_landmark_mean = np.mean(each_landmark, axis=1)
    print('each_landmark_mean: %s' % each_landmark_mean)
    global_mean_nme = np.mean(each_landmark_mean)
    print('mean nme: %s' % global_mean_nme)


def read_annotation(base_dir, label_path):
    """
    :param base_dir: "/mnt/data/changshuang/data/flickr/"
    :param label_path: "/mnt/data/changshuang/data/aflw_anno.txt"
    :return:
    """
    '''读取文件的image, box'''
    data = dict()
    images = []
    bboxes = []
    landmarks = []
    labelfile = open(label_path, 'r')

    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        one_image_landmarks = []
        for i in range(int(nums)):
            # 人脸landmark
            face_landmark = list(map(float, labelfile.readline().strip('\n').split(' ')))
            xmin = face_landmark[0]
            ymin = face_landmark[1]
            xmax = face_landmark[2]
            ymax = face_landmark[3]
            one_image_bboxes.append([xmin, ymin, xmax, ymax, 1])
            # 5 landmark
            lec_x, lec_y = face_landmark[18], face_landmark[19]
            rec_x, rec_y = face_landmark[24], face_landmark[25]
            nc_x, nc_y = face_landmark[32], face_landmark[33]
            mlc_x, mlc_y = face_landmark[38], face_landmark[39]
            mrc_x, mrc_y = face_landmark[42], face_landmark[43]
            one_image_landmarks.append([lec_x, lec_y, rec_x, rec_y, nc_x, nc_y, mlc_x, mlc_y, mrc_x, mrc_y])
        bboxes.append(one_image_bboxes)
        landmarks.append(one_image_landmarks)
    # [[[,,,], ..., [,,,]], [[,,,], ..., [,,,]]]
    data['images'] = images
    data['bboxes'] = bboxes
    data['landmarks'] = landmarks
    return data


def calculate_nme(gt_landmarks, pre_landmarks, gt_bboxes):
    nme = []
    for gt_landmark, pre_landmark, gt_bboxe in zip(gt_landmarks, pre_landmarks, gt_bboxes):
        if 0.0 in gt_landmark:
            continue
        gt_landmark_array = np.array(gt_landmark).reshape(-1, 2)
        pre_landmark_array = pre_landmark.reshape(-1, 2)
        _nme = np.sqrt(np.sum(np.square(gt_landmark_array - pre_landmark_array), axis=1))  # [1,4]
        x1, y1, x2, y2, _ = gt_bboxe
        h = y2 - y1 + 1
        w = x2 - x1 + 1

        _dist = np.sqrt(h * w)
        _nme /= _dist
        nme.append(list(_nme))
    return nme


if __name__ == "__main__":
    evalu = "onet"
    evaluate(evalu)

    # gt_landmarks = np.array([[1, 1, 0, 0, 1, 0, 2, 1, 0, 1], [1, 1, 0, 0, 1, 0, 2, 1, 2, 1]])
    # pre_landmarks = np.array([[1, 2, 0, 2, 1, 1, 2, 0, 1, 1], [1, 0, 0, 1, 0, 1, 2, 2, 0, 1]])
    # bboxe = np.array([[2, 2, 5, 5, 1], [1, 1, 4, 4, 1]])
    # nme = calculate_nme(gt_landmarks, pre_landmarks, bboxe)
    # print(nme)
