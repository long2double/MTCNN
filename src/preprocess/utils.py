import numpy as np
import cv2 as cv
import os

def iou(box, boxes):
    '''
    裁剪的box和图像所有人脸的iou值
    '''
    # box的面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes中每个人脸的面积
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 重叠部分左上右下坐标
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    # 重叠部分长宽
    w = np.maximum(0, x2 - x1 + 1)
    h = np.maximum(0, y2 - y1 + 1)
    # 重叠部分面积
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)


def get_data_from_txt(txt, data_path, name, with_landmark=True):
    '''
    获取txt中的图像路径, 人脸box, 人脸关键点
    参数：
      txt：数据txt文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径,人脸box,关键点)
    '''
    with open(txt, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        # 获取图像路径
        img_path = os.path.join(data_path, components[0]).replace('\\', '/')
        # 人脸box
        if name == 'lfw':
            box = (components[1], components[3], components[2], components[4])
        elif name == 'wider':
            box = (components[1], components[2], components[3], components[4])
        else:
            print('name只能是"lfw"或"wider"')
            exit()
        box = [float(_) for _ in box]
        box = list(map(int, box))

        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue
        # 五个关键点(x,y)
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(components[5 + 2 * index]), float(components[5 + 2 * index + 1]))
            landmark[index] = rv

        # img = cv.imread((img_path))
        # for x, y in landmark:
        #     cv.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        # cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0))
        # cv.imshow('show img', img)
        # if cv.waitKey() == ord('q'):
        #     exit()

        result.append((img_path, BBox(box), landmark))  # [path, BBox Object, [[,], [,], [,], [,], [,]]]
    return result


class BBox:
    # 人脸的box
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        '''将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        '''
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        '''将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        '''
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        '''对所有关键点进行reproject操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        '''对所有关键点进行project操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p


def flip(face,landmark):
    #镜像
    face_flipped_by_x=cv.flip(face,1)
    landmark_=np.asarray([(1-x,y) for (x,y) in landmark])
    landmark_[[0,1]]=landmark_[[1,0]]
    landmark_[[3,4]]=landmark_[[4,3]]
    return (face_flipped_by_x,landmark_)
    

def rotate(img,box,landmark,alpha):
    #旋转
    center=((box.left+box.right)/2,(box.top+box.bottom)/2)
    rot_mat=cv.getRotationMatrix2D(center,alpha,1)
    img_rotated_by_alpha=cv.warpAffine(img,rot_mat,(img.shape[1],img.shape[0]))
    landmark_=np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                         rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x,y) in landmark ])
    face=img_rotated_by_alpha[box.top:box.bottom+1,box.left:box.right+1]
    return (face,landmark_)


def read_annotation(base_dir, label_path):
    """
    base_dir: ../data/WIDER_train/
    label_path: ../data/wider_face_train_bbx_gt.txt
    """
    '''读取文件的image, box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')

    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')
        if nums == '0':
            images.pop()
            bb_info = labelfile.readline().strip('\n').split(' ')
            continue
     
        one_image_bboxes = []
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            # 人脸框
            face_box = [float(bb_info[i]) for i in range(4)]
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2] - 1
            ymax = ymin + face_box[3] - 1
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)
    # [[[,,,], ..., [,,,]], [[,,,], ..., [,,,]]]
    data['images'] = images
    data['bboxes'] = bboxes
    return data


class TestLoader:
    # 制作迭代器
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)
        self.cur = 0
        self.data = None
        self.label = None
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.imdb)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        imdb = self.imdb[self.cur]
        im = cv.imread(imdb)
        self.data = im


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
