import os
import random
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm
from utils import iou, get_data_from_txt, BBox, flip, rotate
import argparse


def rebuild_file():
    anno_file = '/mnt/data/changshuang/data/label.txt'
    save_file = '/mnt/data/changshuang/data/wider_face_train_landmark.txt'

    f = open(anno_file, 'r')
    f1 = open(save_file, 'w')

    while True:
        image_line = f.readline().strip().split(' ')
        if image_line[0] == '':
            break
        if image_line[0] == '#':
            image_path = image_line[1]
        else:
            if image_line.count('-1.0') >= 1 or image_line[-1] <= str(0.2):
                continue
            x, y, w, h = list(map(int, image_line[:4]))
            if x < 0 or y < 0 or h < 20 or w < 20:
                continue
            bbox = list(map(str, [x, y, x + w - 1, y + h - 1]))
            landmark = np.array(image_line[4:-1]).reshape(-1, 3)[:, :2]
            landmark = list(landmark.reshape(-1))
            line_information = image_path + ' ' + ' '.join(bbox) + ' ' + ' '.join(landmark) + '\n'
            f1.write(line_information)
    f.close()
    f1.close()


def gen_lfw_landmark(img_dir, save_dir, input_size, argument, name):
    # 数据输出路径
    base_dir = os.path.join(save_dir, str(input_size))
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    landmark_save_dir = os.path.join(base_dir, '%s_landmark' % name)
    if not os.path.exists(landmark_save_dir):
        os.mkdir(landmark_save_dir)

    # label记录txt
    if name == 'lfw':
        ftxt = os.path.join(img_dir, 'trainImageList.txt')
    elif name == 'wider':
        ftxt = os.path.join(img_dir, 'wider_face_train_landmark.txt')
        img_dir = os.path.join(img_dir, 'WIDER_train/images')
    else:
        print('name只能是"lfw"或"wider"')
        exit()
    # 记录label的txt
    f = open(os.path.join(base_dir, '%s_landmark.txt' % name), 'w')
    # 获取图像路径, box, 关键点
    data = get_data_from_txt(ftxt, img_dir, name)  # lfw data format: [(path, BBox object, [[,], [,], [,], [,], [,]]), ]

    idx = 0
    image_id = 0
    for (imgPath, box, landmarkGt) in tqdm(data):
        # 存储人脸图片和关键点
        F_imgs = []
        F_landmarks = []
        img = cv.imread(imgPath)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([box.left, box.top, box.right, box.bottom])
        # 人脸图片
        f_face = img[box.top:box.bottom + 1, box.left:box.right + 1]
        # resize成网络输入大小
        f_face = cv.resize(f_face, (input_size, input_size))
        
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
            # 关键点相对于左上坐标偏移量并归一化
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))

        landmark = np.zeros((5, 2))
        if argument:
            # 对图像变换
            idx = idx+1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            # 除去过小图像
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(10):
                # 随机裁剪图像大小
                box_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # 随机左上坐标偏移量
                delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
                # 计算左上坐标
                nx1 = int(max(x1 + gt_w/2 - box_size/2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h/2 - box_size/2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                # 除去超过边界的
                if nx2 > img_w or ny2 > img_h:
                    continue
                # 裁剪边框, 图片
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2+1, nx1:nx2+1, :]
                resized_im = cv.resize(cropped_im, (input_size, input_size))
                Iou=iou(crop_box, np.expand_dims(gt_box, 0))
                #只保留pos图像
                if Iou > 0.65:
                    F_imgs.append(resized_im)
                    #关键点相对偏移
                    for index,one in enumerate(landmarkGt):
                        rv=((one[0]-nx1)/box_size,(one[1]-ny1)/box_size)
                        landmark[index]=rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark=np.zeros((5,2))
                    landmark_=F_landmarks[-1].reshape(-1,2)
                    box=BBox([nx1,ny1,nx2,ny2])
                    #镜像
                    if random.choice([0,1])>0:
                        face_flipped,landmark_flipped=flip(resized_im,landmark_)
                        face_flipped=cv.resize(face_flipped,(input_size,input_size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #逆时针翻转
                    if random.choice([0,1])>0:
                        face_rotated_by_alpha,landmark_rorated=rotate(img,box, box.reprojectLandmark(landmark_),5)
                        #关键点偏移
                        landmark_rorated=box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha=cv.resize(face_rotated_by_alpha,(input_size,input_size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))
                        
                        #左右翻转
                        face_flipped,landmark_flipped=flip(face_rotated_by_alpha,landmark_rorated)
                        face_flipped=cv.resize(face_flipped,(input_size,input_size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #顺时针翻转
                    if random.choice([0,1])>0:
                        face_rotated_by_alpha,landmark_rorated=rotate(img,box, box.reprojectLandmark(landmark_),-5)
                        #关键点偏移
                        landmark_rorated=box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha=cv.resize(face_rotated_by_alpha,(input_size,input_size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))
                        
                        #左右翻转
                        face_flipped,landmark_flipped=flip(face_rotated_by_alpha,landmark_rorated)
                        face_flipped=cv.resize(face_flipped,(input_size,input_size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

        for i in range(len(F_imgs)):
            # 剔除数据偏移量在[0,1]之waide
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            cv.imwrite(os.path.join(landmark_save_dir, '%d.jpg' % (image_id)), F_imgs[i])
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(landmark_save_dir, '%d.jpg' % (image_id)) + ' -2 ' + ' '.join(landmarks)+'\n')
            image_id += 1
    f.close()


if __name__ == '__main__':
    # rebuild_file()
    parser = argparse.ArgumentParser(description='gen_lfw_landmark')
    parser.add_argument('--img_dir', type=str, default='/mnt/data/changshuang/data',
                        help='图片路径')
    parser.add_argument('--save_dir', type=str, default='/mnt/data/changshuang/gen_data',
                        help='保存路径')
    parser.add_argument('--name', type=str, default='wider',
                        help='选择landmark数据集')
    parser.add_argument('--input_size', type=int, default=12,
                        help='对于具体网络输入图片的大小')
    parser.add_argument('--argument', type=bool, default=True,
                        help='是否对图像进行变换')
    args = parser.parse_args()
    gen_lfw_landmark(args.img_dir, args.save_dir, args.input_size, args.argument, args.name)
