import numpy as np
import argparse
import pickle
import sys
import os
from tqdm import tqdm
import cv2 as cv
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from network.model import p_net, r_net, o_net
from detection.mtcnn_detector import MtcnnDetector
from detection.detector import Detector, PNetDetector
import config
from preprocess.utils import read_annotation, TestLoader, convert_to_square, iou


root_dir = os.path.dirname(__file__).split('MTCNN')[0]
project_dir = os.path.dirname(__file__).split('MTCNN')[1]


def gen_hard_example(img_dir, save_dir, input_size):
    '''通过pnet或rnet生成下一个网络的输入'''
    size = input_size
    batch_size = config.batches
    min_face_size = config.min_face
    stride = config.stride
    thresh = config.thresh
    # 模型地址
    model_path = [root_dir + 'MTCNN/checkpoint/pnet/pnet-1', root_dir + 'MTCNN/checkpoint/rnet/rnet-1', root_dir + 'MTCNN/checkpoint/onet/onet-1']
    net, save_size = None, None
    if input_size == '12':
        net = 'pnet'
        save_size = 24
    elif input_size == '24':
        net = 'rnet'
        save_size = 48
    assert net is not None and size is not None
    # 图像数据地址
    wider_img_dir = os.path.join(img_dir, 'WIDER_train')
    # 处理后的图像存放地址
    data_dir = os.path.join(save_dir, str(save_size))
    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(neg_dir):
        os.mkdir(neg_dir)  
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    if not os.path.exists(part_dir):
        os.mkdir(part_dir) 
    
    detectors = [None, None, None]
    pnet = PNetDetector(p_net, model_path[0])
    detectors[0] = pnet

    if net == 'rnet':
        rnet = Detector(r_net, 24, batch_size[1], model_path[1])
        detectors[1] = rnet
    
    mtcnn_detector = MtcnnDetector(detectors, min_face_size, stride, thresh)
    anno_file = os.path.join(img_dir, 'wider_face_train_bbx_gt.txt')
    # 读取wider face文件的image和bbox
    data = read_annotation(wider_img_dir, anno_file)
    # data: {'images': images, 'bboxes': bboxes}
    # bboxes: [[[,,,], ..., [,,,]], [[,,,], ..., [,,,]]]
    # 将data制作成迭代器, input image path, output image data
    print('载入数据')
    test_data = TestLoader(data['images'])
    detectors, _ = mtcnn_detector.detect_face(test_data)
    print('完成识别')

    save_file = os.path.join(data_dir, 'detections.pkl')
    # save to save_file
    with open(save_file, 'wb') as f:
        pickle.dump(detectors, f, 1)
    print('开始生成图像')
    save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, data_dir)


def save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, data_dir):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    # save files
    neg_label_file = os.path.join(data_dir, "neg_%d.txt" % save_size)
    neg_file = open(neg_label_file, 'w')
    pos_label_file = os.path.join(data_dir, "pos_%d.txt" % save_size)
    pos_file = open(pos_label_file, 'w')
    part_label_file = os.path.join(data_dir, "part_%d.txt" % save_size)
    part_file = open(part_label_file, 'w')
    # read detect result
    det_boxes = pickle.load(open(os.path.join(data_dir, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
    assert len(det_boxes) == num_of_images, "弄错了"
    
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    
    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv.imread(im_idx)
        # 转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
           
            Iou = iou(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv.resize(cropped_im, (save_size, save_size), interpolation=cv.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                neg_file.write(save_file + ' 0\n')
                cv.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen_hard_example')
    parser.add_argument('--img_dir', type=str, default='/mnt/data/changshuang/data',
                        help='图片路径')
    parser.add_argument('--save_dir', type=str, default='/mnt/data/changshuang/gen_data',
                        help='保存路径')
    parser.add_argument('--input_size', type=str, default='24',
                        help='对于具体网络输入图片的大小')
    args = parser.parse_args()
    gen_hard_example(args.img_dir, args.save_dir, args.input_size)
    
