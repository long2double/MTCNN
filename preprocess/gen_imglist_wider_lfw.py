# coding: utf-8
import numpy as np
import os
import argparse
import sys

npr = np.random
data_dir = '/mnt/data/changshuang/data/'


def main(args):
    '''将pos,part,neg,landmark四者混在一起'''
    size = args.input_size
    if size == 12:
        net = 'PNet'
    elif size == 24:
        net = 'RNet'
    elif size == 48:
        net = 'ONet'
    outdir = data_dir + '{}'.format(size)
    print(outdir)
    with open(os.path.join(outdir, 'wider_landmark_%d_aug.txt' % (size)), 'r') as f:
        wider_face = f.readlines()
    with open(os.path.join(outdir, 'landmark_%d_aug.txt' % (size)), 'r') as f:
        lfw = f.readlines()

    # merge dataset save to train_pnet_landmark.txt
    with open(os.path.join(outdir, 'train_wider_lfw_landmark_aug.txt'), 'w') as f:
        lfw_keep = npr.choice(len(lfw), size=len(lfw), replace=True)
        wider_face_keep = npr.choice(len(wider_face), size=len(wider_face), replace=False)  # put back the sample
        for i in lfw_keep:
            f.write(lfw[i])
        for i in wider_face_keep:
            f.write(wider_face[i])
        print('wider_face数量：{} lfw数量：{}'.format(len(wider_face), len(lfw)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
