import numpy as np
import argparse
import os


def gen_pnet_imglist(save_file, input_size, base_num, name):
    base_dir = os.path.join(save_file, input_size)
    with open(os.path.join(base_dir, 'pos_12.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(base_dir, 'neg_12.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(base_dir, 'part_12.txt'), 'r') as f:
        part = f.readlines()
    if name == 'lfw':
        anno_file = 'lfw_landmark.txt'
    elif name == 'wider':
        anno_file = 'wider_landmark.txt'
    elif name == 'lfw_wider':
        anno_file = 'lfw_wider_landmark.txt'
    with open(os.path.join(base_dir, anno_file), 'r') as f:
        landmark = f.readlines()
    print('neg数量：{} pos数量：{} part数量:{} landmark数量:{}'.format(len(neg), len(pos), len(part), len(landmark)))

    with open(os.path.join(base_dir, 'pnet_landmark.txt'), 'w') as f:
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=True)  # put back the sample
        sum_p = len(neg_keep) // 3
        pos_keep = np.random.choice(len(pos), sum_p, replace=True)
        part_keep = np.random.choice(len(part), sum_p, replace=True)
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for item in landmark:
            f.write(item)
        print('keep neg数量：{} keep pos数量：{} keep part数量:{}, landmark数量:{}, 基数:{}'.format(len(neg_keep), len(pos_keep), len(part_keep), len(landmark), base_num))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='gen_pnet_imglist')
    parse.add_argument('--save_file', type=str, default='/mnt/data/changshuang/gen_data',
                        help='保存图片')
    parse.add_argument('--input_size', type=str, required=True, choices=['12', '24', '48'],
                        help='对于具体网络输入图片的大小')
    parse.add_argument('--base_num', type=int, default=300000,
                        help='neg基础数量')
    parse.add_argument('--name', type=str, required=True, choices=['lfw', 'wider', 'lfw_wider'],
                        help='landmark是否含有wider数据集')
    args = parse.parse_args()
    gen_pnet_imglist(args.save_file, args.input_size, args.base_num, args.name)
