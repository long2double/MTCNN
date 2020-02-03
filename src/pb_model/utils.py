import numpy as np


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


def convert_to_square(box):
    '''
    将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box,[n,5]
    '''
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def pad(bboxes, w, h):
    '''
    将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : 为调整后的box右下角相对调整后的box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    '''
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]
    return return_list

