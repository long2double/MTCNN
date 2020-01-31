import numpy as np
import argparse
import tensorflow as tf
import cv2 as cv
import sys
import os
from tqdm import tqdm
import random


def gen_tfrecords(save_file, input_size, name):
    base_dir = os.path.join(save_file, input_size)
    tfrecord_dir = os.path.join(base_dir, 'tfrecord')
    if not os.path.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)
    
    # pnet只生成一个混合的tfrecords, rnet和onet要分别生成4个
    if input_size == '12':
        tf_filenames = [os.path.join(tfrecord_dir, 'pnet.tfrecord')]
        items = ['pnet_landmark.txt']
    else:
        tf_filename1 = os.path.join(tfrecord_dir, 'pos_%s.tfrecord' % input_size)
        item1 = 'pos_%s.txt' % input_size
        tf_filename2 = os.path.join(tfrecord_dir, 'part_%s.tfrecord' % input_size)
        item2 = 'part_%s.txt' % input_size
        tf_filename3 = os.path.join(tfrecord_dir, 'neg_%s.tfrecord' % input_size)
        item3 = 'neg_%s.txt' % input_size
        tf_filename4 = os.path.join(tfrecord_dir, 'landmark_%s.tfrecord' % input_size)
        if name == 'lfw':
            item4 = 'lfw_landmark.txt' 
        elif name == 'wider':
            item4 = 'lfw_wider_landmark.txt'
        tf_filenames = [tf_filename1, tf_filename2, tf_filename3, tf_filename4]
        items = [item1, item2, item3, item4]
    if tf.gfile.Exists(tf_filenames[0]):
        print('tfrecords文件早已生成，无需此操作')
        return
    # 获取数据
    for tf_filename, item in zip(tf_filenames, items):
        print('开始读取数据')
        """
        for imglist.txt processing -->>> dict
        [data_example1, data_example2 ....]
        data_example {'filename': path, 'label': label, 'bbox' bbox}
        bbox {'xmin':,'ymin':....}
        """
        dataset = get_dataset(base_dir, item)
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)
        print('开始转换tfrecords')
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for image_example in tqdm(dataset):
                filename = image_example['filename']
                try:
                    _add_to_tfrecord(filename, image_example, tfrecord_writer)
                except:
                    print(filename)
    print('完成转换')

def get_dataset(base_dir, item):
    '''
    从txt获取数据
    参数：
      dir：存放数据目录
      item:txt目录
    返回值：
      包含label,box, 关键点的data
    '''
    dataset_dir = os.path.join(base_dir, item)
    imagelist = open(dataset_dir, 'r')
    dataset=[]
    for line in tqdm(imagelist.readlines()):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]  # path ../data/24/negative/0.jpg
        data_example['label'] = int(info[1])  # label 0, -1, 1 -2
        # neg的box默认为0,part,pos的box只包含人脸框, landmark的box只包含关键点
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset
    # [data_example1, data_example2 ....]
    # data_example {'filename': path, 'label': label, 'bbox' bbox}
    # bbox {'xmin':,'ymin':....}


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    '''
    转换成tfrecord文件
    参数：
      filename：图片文件名
      image_example:数据
      tfrecord_writer:写入文件
    '''
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _process_image_withoutcoder(filename):
    '''读取图片文件,返回图片大小'''
    image = cv.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


# 不同类型数据的转换
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example_simple(image_example, image_buffer):
    '''转换成tfrecord接受形式'''
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='gen_tfrecords')
    parse.add_argument('--save_file', type=str, default='/mnt/data/changshuang/gen_data',
                        help='保存图片路径')
    parse.add_argument('--input_size', type=str, required=True, choices=['12', '24', '48'],
                        help='对于具体网络输入图片的大小')
    parse.add_argument('--name', type=str, required=True, choices=['lfw', 'wider'],
                        help='landmark是否包含wider数据集')
    args = parse.parse_args()
    gen_tfrecords(args.save_file, args.input_size, args.name)
    
