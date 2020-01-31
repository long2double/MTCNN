# coding: utf-8
from detection.mtcnn_detector import MtcnnDetector
from detection.detector import Detector, PNetDetector
from network.model import p_net, r_net, o_net
import cv2 as cv
import os
import sys
import config
import tensorflow as tf
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test_lfw(img_dir, anno_file):
    thresh = config.thresh
    min_face_size = config.min_face
    stride = config.stride
    batch_size = config.batches
    detectors = [None, None, None]

    # 模型放置位置
    model_path = ['checkpoint/pnet/pnet-30', 'checkpoint/rnet/rnet-22', 'checkpoint/onet/onet-30']

    detectors[0] = PNetDetector(p_net, model_path[0])
    detectors[1] = Detector(r_net, 24, batch_size[1], model_path[1])
    detectors[2] = Detector(o_net, 48, batch_size[2], model_path[2])

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh)

    labelfile = open(anno_file, 'r')
    while True:
        imagepath = labelfile.readline().replace('\\', '/').strip('\n').split(' ')
        if not imagepath:
            break
        path = imagepath[0]
        bboxs = imagepath[1:5]
        landmarks = imagepath[5:]

        imagepath = os.path.join(img_dir, path)
        img = cv.imread(imagepath)
        boxes_c, landmarks = mtcnn_detector.detect(img)
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            cv.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            # 判别为人脸的置信度
            cv.putText(img, '{:.2f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 画关键点
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv.circle(img, (int(landmarks[i][2*j]), int(int(landmarks[i][2*j+1]))), 2, (0, 0, 255), -1)
        cv.imshow('im', img)
        if cv.waitKey() & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test_lfw')
    parser.add_argument('--img_dir', type=str, default='/mnt/data/changshuang/data',
                        help='图像路径')
    parser.add_argument('--anno_file', type=str, default='/mnt/data/changshuang/data/trainImageList.txt',
                        help='lfw测试图像信息')
    args = parser.parse_args()
    test_lfw(args.img_dir, args.anno_file)

