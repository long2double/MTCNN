import numpy as np
import os
import argparse
import sys

def gen_lfw_wider_imglist(save_file, input_size):
    base_dir = os.path.join(save_file, input_size)
    with open(os.path.join(base_dir, 'lfw_landmark.txt'), 'r') as f:
        lfw_anno = f.readlines()
    with open(os.path.join(base_dir, 'wider_landmark.txt'), 'r') as f:
        wider_anno = f.readlines()

    with open(os.path.join(base_dir, 'lfw_wider_landmark.txt'), 'w') as f:
        lfw_keep = np.random.choice(len(lfw_anno), size=len(lfw_anno), replace=False)  # 不放回
        wider_keep = np.random.choice(len(wider_anno), size=len(wider_anno), replace=False)
        for i in lfw_keep:
            f.write(lfw_anno[i])
        for i in wider_keep:
            f.write(wider_anno[i])
    print('lfw_landmark的数量：{}, wider_face_landmark的数量：{}'.format(len(lfw_anno), len(wider_anno)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen_lfw_wider_imglist')
    parser.add_argument('--save_file', type=str, default='/mnt/data/changshuang/gen_data',
                        help='图片路径')
    parser.add_argument('--input_size', type=str, required=True, choices=['12', '24', '48'],
                        help='对于具体网络输入图片的大小')
    args = parser.parse_args()
    gen_lfw_wider_imglist(args.save_file, args.input_size)
    