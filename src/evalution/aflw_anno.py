import sqlite3
import cv2 as cv
import numpy as np
import os
import argparse
from itertools import chain


def aflw_anno(sqlite_path, aflw_anno):
    anno_format = "{} {} {} {} {} {}"
    conn = sqlite3.connect(sqlite_path)
    face_ids = conn.execute("SELECT face_id FROM faces")
    # 保存的数据集 {"image_path1":[count,[,...,]]，..., "image_path2":[count,[,...,]]}
    anno_dict = {} 
    num = 0
    for face_id in face_ids:
        # print("当前人脸的id:", face_id[0])
        # 当前人脸多对应的图片id
        img_ids = conn.execute("SELECT file_id FROM Faces WHERE face_id={}".format(face_id[0]))
        img_ids = [img_id for img_id in img_ids]  # [('image65729.jpg',)]

        # 图片的信息，路径，宽，高
        fileID = conn.execute("SELECT filepath, width, height FROM FaceImages WHERE file_id='{}'".format(img_ids[0][0]))
        fileID = [id for id in fileID]  # [('0/image65668.jpg', 500, 375)]
        file_path = fileID[0][0]
        width = fileID[0][1]
        height = fileID[0][2]
        # print("当前人脸所对应的图片路径,宽,高:", file_path, width, height)

        # bbox
        faceRect = conn.execute("SELECT x, y, w, h FROM faceRect WHERE face_id={}".format(face_id[0]))
        faceRect = [id for id in faceRect]
        if len(faceRect) == 0:
            """
            65385 65386 65387 65388 65389 65390 65391 65392 65393 65394 65395 65396
            数据集中的数据在图片集中没有找到
            """
            continue
        num += 1
        x1, y1, w, h = faceRect[0]
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height
        # print("当前人脸所对应的bbox左上,右下坐标:", x1, y1, x2, y2)

        # 21个landmark
        landmark_coord_list = [[0.0, 0.0] for _ in range(21)]
        feature_ids = conn.execute("SELECT feature_id, x, y FROM featurecoords WHERE face_id='{}'".format(face_id[0]))
        feature_ids = [(id[0], round(id[1], 1), round(id[2], 1)) for id in feature_ids]
        for index, x, y in feature_ids:
            landmark_coord_list[index - 1][0] = x
            landmark_coord_list[index - 1][1] = y
        landmark_coord_list = ' '.join(map(str, list(chain.from_iterable(landmark_coord_list))))
        # print("当前人脸所对应的21个landmark坐标:", landmark_coord_list)

        # 每一行完整的人脸数据，img_path，x1,y1,x2,y2,21个landmarks坐标点
        anno_line = anno_format.format(file_path, x1, y1, x2, y2, landmark_coord_list)

        # 转换为dict,格式为{"image_path1":[count,[,...,]]，..., "image_path2":[count,[,...,]]}
        anno_list = anno_line.split()
        if anno_list[0] not in anno_dict:
            value = [0,[]]
            value[0] = 1
            value[1].append(' '.join(anno_list[1:]))
            anno_dict[anno_list[0]] = value
        else:
            value = anno_dict[anno_list[0]]
            value[0] += 1
            value[1].append(' '.join(anno_list[1:]))

    """
    数据格式与WiderFace类似
    第一行当前图片的路径
    第二行当前图片中有n个人脸
    接下来的n行，为当前图片所对应的n个人脸，每个人脸包含的bbox和21个landmark
    """
    with open(aflw_anno, "w") as f:
        for key, value in anno_dict.items():
            f.writelines("%s\n" % key)
            f.writelines("%s\n" % value[0])
            f.writelines("%s\n" % line for line in value[1])
    f.close()
    print("一共有%d个人脸被保存" % num)

def show(anno_file, aflw_dir, output):
    f = open(anno_file, "r")
    img_num = 0	
    while True:
        image_path = f.readline().strip("\n")
        if not image_path:
            break
        img = cv.imread(os.path.join(aflw_dir, image_path))
        count = int(f.readline().strip("\n"))
        for _ in range(count):
            coord_list = f.readline().strip("\n").split(" ")
            x1, y1, x2, y2 = list(map(int, coord_list[:4]))
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

            landmarks_coord = np.array(list(map(float, coord_list[4:]))).reshape(-1, 2)
            for i in range(landmarks_coord.shape[0]):
                x = landmarks_coord[i][0]
                y = landmarks_coord[i][1]
                if x != 0 or y != 0:
                    cv.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
        if not os.path.exists(output):
            os.mkdir(output)
        cv.imwrite(output + '/{:0>6}.jpg'.format(img_num), img)
        img_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aflw_anno')
    parser.add_argument('--sqlite_path', type=str, default='/mnt/data/changshuang/data/aflw.sqlite',
                        help='aflw数据库sqlite的路径')
    parser.add_argument('--aflw_anno', type=str, default='/mnt/data/changshuang/data/aflw_anno.txt',
                        help='aflw数据库读取数据后保存的文件路径')
    parser.add_argument('--aflw_dir', type=str, default='/mnt/data/changshuang/data/flickr', 
                        help='aflw图像保存的路径')
    parser.add_argument('--output', type=str, default='/mnt/data/changshuang/gen_data/output_aflw',
                        help='验证aflw数据集读取是否正确')
    args = parser.parse_args()
    # aflw_anno(args.sqlite_path, args.aflw_anno)
    show(args.aflw_anno, args.aflw_dir, args.output)
