import numpy as np
import os
import cv2 as cv
from tqdm import tqdm
from utils import iou
import argparse


def gen_pnet_data(anno_file, img_file, save_file):
    # pos,part,neg裁剪图片存放位置
    base_dir = os.path.join(save_file, '12')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    pos_save_dir = os.path.join(base_dir, 'positive')
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)

    part_save_dir = os.path.join(base_dir, 'part')
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)

    neg_save_dir = os.path.join(base_dir, 'negative')
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    # pos,part,neg对应的anno_file
    f1 = open(os.path.join(base_dir, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(base_dir, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(base_dir, 'part_12.txt'), 'w')
    # wider_face_train对应的anno_file
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print('总共的图片数：%d' % num)

    # 记录pos,neg,part三类生成数
    p_idx = 0
    n_idx = 0
    d_idx = 0
    # 记录读取图片数
    idx = 0
    for annotation in tqdm(annotations):
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        box = list(map(float, annotation[1:]))
        
        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)
        
        img = cv.imread(os.path.join(img_file, im_path+'.jpg'))
        idx += 1
        height, width, channel = img.shape
        
        neg_num = 0
        # 先采样一定数量neg图片
        while neg_num < 50:
            # 随机选取截取图像大小
            size = np.random.randint(12, min(width, height)/2)
            # 随机选取左上坐标
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            # 截取box
            crop_box = np.array([nx, ny, nx+size, ny+size])
            # 计算iou值
            Iou = iou(crop_box, boxes)
            # 截取图片并resize成12x12大小
            cropped_im = img[ny:ny + size, nx:nx + size, :]
            resized_im = cv.resize(cropped_im, (12, 12), interpolation=cv.INTER_LINEAR)
            # iou值小于0.3判定为neg图像
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write(neg_save_dir + '/%s.jpg' % n_idx+' 0\n')
                cv.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
        for box in boxes:
            # 左上右下坐标
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            # 舍去图像过小和box在图片外的图像
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            for i in range(5):
                size = np.random.randint(12, min(width, height)/2)
                # 随机生成的关于x1,y1的偏移量，并且保证x1+delta_x>0,y1+delta_y>0
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                # 截取后的左上角坐标
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # 排除大于图片尺度的
                if nx1 + size > width or ny1+size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = iou(crop_box, boxes)
                cropped_im = img[ny1:ny1 + size, nx1:nx1 + size, :]
                resized_im = cv.resize(cropped_im, (12, 12), interpolation=cv.INTER_LINEAR)
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                    f2.write(neg_save_dir + '/%s.jpg' % n_idx + ' 0\n')
                    cv.imwrite(save_file, resized_im)
                    n_idx += 1
            for i in range(20):
                # 缩小随机选取size范围，更多截取pos和part图像
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # 除去尺度小的
                if w < 5:
                    continue
                # 偏移量, 范围缩小了
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                # 截取图像左上坐标计算是先计算x1+w/2表示的中心坐标,再+delta_x偏移量,再-size/2, 变成新的左上坐标
                nx1 = int(max(x1 + w/2 + delta_x - size/2, 0))
                ny1 = int(max(y1 + h/2 + delta_y - size/2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                
                # 排除超出的图像
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                # 人脸框相对于截取图片的偏移量并做归一化处理
                offset_x1 = (x1 - nx1)/float(size)
                offset_y1 = (y1 - ny1)/float(size)
                offset_x2 = (x2 - nx2)/float(size)
                offset_y2 = (y2 - ny2)/float(size)
                
                cropped_im = img[ny1:ny2, nx1:nx2, :]
                resized_im = cv.resize(cropped_im, (12, 12), interpolation=cv.INTER_LINEAR)
                # box扩充一个维度作为iou输入
                box_ = box.reshape(1, -1)
                Iou = iou(crop_box, box_)
                if Iou >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write(pos_save_dir + '/%s.jpg' % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    p_idx += 1
                elif Iou >= 0.4:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write(part_save_dir+'/%s.jpg' % d_idx+' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file,resized_im)
                    d_idx += 1

    print('%s 个图片已处理，pos：%s  part: %s neg:%s' % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen_pnet_data')
    parser.add_argument('--anno_file', type=str, default='/mnt/data/changshuang/data/wider_face_train.txt',
                        help='注释路径')
    parser.add_argument('--img_file', type=str, default='/mnt/data/changshuang/data/WIDER_train/images', 
                        help='图片路径')
    parser.add_argument('--save_file', type=str, default='/mnt/data/changshuang/gen_data', 
                        help='保存路径')
    args = parser.parse_args()
    gen_pnet_data(args.anno_file, args.img_file, args.save_file)

