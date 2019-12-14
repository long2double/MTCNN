# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import numpy.random as npr
from src.utils.iou import IOU
from utils import logger

parser = argparse.ArgumentParser(description="generate pnet data")
parser.add_argument("--anno_file", type=str,
                    default="../../../data/WIDER_FACE/wider_face_train.txt",
                    help="wider face dataset annotation file")
parser.add_argument("--img_dir", type=str,
                    default="../../../data/WIDER_FACE/WIDER_train/images",
                    help="wider face dataset image path")
args = parser.parse_args()


def gen_pnet_data(anno_file, img_dir):
    log = logger.setup_logger('../../logs/%s.log' % __file__.split('/')[-1])
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ": " + str(value))

    pos_save_dir = "../../../total_data/pnet/detection/pos"  # generate pnet's positive sample save path
    part_save_dir = "../../../total_data/pnet/detection/part"  # generate pnet's part sample save path
    neg_save_dir = "../../../total_data/pnet/detection/neg"  # generate pnet's negative sample save path
    annos_dir = "../../../total_data/pnet/detection"  # generate pnet's positive,part,negative sample correlation annotation file
    if not os.path.exists(annos_dir):
        os.makedirs(annos_dir)
    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.makedirs(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)

    pos_file = open(os.path.join(annos_dir, 'pos_12.txt'), 'w')
    neg_file = open(os.path.join(annos_dir, 'neg_12.txt'), 'w')
    part_file = open(os.path.join(annos_dir, 'part_12.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    log.info("wider_face_train文件中图片/注释的个数是： %d" % num)

    p_idx = 0  # 正样例个数
    n_idx = 0  # 负样例个数
    d_idx = 0  # 部分样例个数
    # 数据集中bbox的个数
    bbox_num = 0
    for index, annotation in enumerate(annotations):
        annotation = annotation.strip().split(' ')
        # 相对于WIDER_train/images路径下的文件路径
        im_path = annotation[0]
        # 图片的多个bbox,左,上,右,下坐标值
        bbox = list(map(float, annotation[1:]))
        # 多个bbox转换为[None, 4]的二维数组
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        # 获得图片的整个地址,并读取图片
        img = cv2.imread(os.path.join(img_dir, im_path + '.jpg'))
        height, width, channel = img.shape
        # 每张图片中生成负样例的计数变量
        neg_num = 0
        # 从每张图像上随机采样50个负样例
        while neg_num < 50:
            # 负样例的尺寸是从12到min(width, height)/2的均匀分布
            size = npr.randint(12, min(width, height) / 2)
            # 左侧点的坐标值
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            # 随机采样到的bbox,[nx, ny, nx + size, ny + size]左,上,右,下坐标值
            crop_box = np.array([nx, ny, nx + size, ny + size])
            # 计算采样到的bbox与注释中所有bbox的iou
            iou = IOU(crop_box, boxes)
            # 只有采样到的bbox与所有真实的bbox的iou小于0.3才作为负样例
            if np.max(iou) < 0.3:
                # 图片上的随机采样到bbox的ROI
                cropped_im = img[ny: ny + size, nx: nx + size, :]
                # 将ROI调整到12×12大小
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                # 将负样例保存的路径写入neg_12.txt中
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)  # '../../OutPut/neg/xxx.jpg'
                neg_file.write(save_file + ' 0\n')
                # 保存负样例的图片
                cv2.imwrite(save_file, resized_im)
                # 总的负样例个数加1,当前图片的负样例加1（每个图片采集50个）
                n_idx += 1
                neg_num += 1

        # 遍历每个bbox
        for box in boxes:
            bbox_num += 1
            # bbox的左,上,右,下坐标值
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            # 考虑到小脸的真实bbox是不精确的,因此忽略正式bbox中小脸的部分,另外也忽略了左上坐标为负的情况
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            # 也额外在bbox附近采样5个负样例,其标准也是iou小于0.3
            neg_num = 0
            while neg_num < 5:
                # size of the image to be cropped
                size = npr.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                # max here not really necessary
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
                if nx1 + size > width or ny1 + size > height:
                    continue
                # 随机采样到的bbox,[nx, ny, nx + size, ny + size]左,上,右,下坐标值
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                # 这里是与当前图像所有的bbox求iou
                iou = IOU(crop_box, boxes)
                if np.max(iou) < 0.3:
                    # 图片上的随机采样到bbox的ROI
                    cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                    # 将ROI调整到12×12大小
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    # 将负样例保存的路径写入neg_12.txt中
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)  # '../../OutPut/neg/xxx.jpg'
                    neg_file.write(save_file + ' 0\n')
                    # 保存负样例的图片
                    cv2.imwrite(save_file, resized_im)
                    # 总的负样例个数加1,当前图片的负样例加1（每个图片采集50个）
                    n_idx += 1
                    neg_num += 1

            # 生成正样例和部分样例
            for i in range(20):
                # 正样例和部分样例尺寸[minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # delta here is the offset of box center
                if w < 5:
                    # print (w)
                    continue
                # print (box)
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                # show this way: nx1 = max(x1+w/2-size/2+delta_x)
                # x1+ w/2 is the central point, then add offset , then deduct size/2
                # deduct size/2 to make sure that the right bottom corner will be out of
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                # show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                # 随机采样到的bbox和真实的bbox的偏移率作为标签值
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                box_ = box.reshape(1, -1)
                iou = IOU(crop_box, box_)

                if iou >= 0.65:
                    cropped_im = img[ny1: ny2, nx1: nx2, :]
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)  # "../../OutPut/pos/xxx.jpg"
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif iou >= 0.4:
                    cropped_im = img[ny1: ny2, nx1: nx2, :]
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)  # "../../OutPut/part/xxx.jpg"
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
        if index % 100 == 0:
            log.info("第{:0>5}张图片完成采样,\tbbox个数: {:0>7},\t pos个数: {:0>7},\t part个数: {:0>7},\t neg个数: {:0>7}".format(index + 1, bbox_num, p_idx, d_idx, n_idx))
    log.info("从{:0>5}张图片进行采样数据,\t一共有bbox个数： {:0>7},\t 一共有pos个数: {:0>7},\t part个数: {:0>7},\t neg个数: {:0>7}".format(index + 1, bbox_num, p_idx, d_idx, n_idx))
    pos_file.close()
    neg_file.close()
    part_file.close()


if __name__ == "__main__":
    gen_pnet_data(args.anno_file, args.img_dir)
