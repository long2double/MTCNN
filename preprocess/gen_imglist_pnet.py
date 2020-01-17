# coding: utf-8
import numpy as np
import os

npr = np.random
data_dir = '/mnt/data/changshuang/data/'


'''将pos,part,neg,landmark四者混在一起'''
size = 12
with open(os.path.join(data_dir, '12/pos_12.txt'), 'r') as f:
    pos = f.readlines()
with open(os.path.join(data_dir, '12/neg_12.txt'), 'r') as f:
    neg = f.readlines()
with open(os.path.join(data_dir, '12/part_12.txt'), 'r') as f:
    part = f.readlines()
with open(os.path.join(data_dir, '12/train_wider_lfw_landmark_aug.txt'), 'r') as f:
    landmark = f.readlines()
print('neg数量：{} pos数量：{} part数量:{} landmark数量:{}'.format(len(neg), len(pos), len(part), len(landmark)))

dir_path = os.path.join(data_dir, '12')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# merge dataset save to train_pnet_landmark.txt
with open(os.path.join(dir_path, 'train_pnet_landmark.txt'), 'w') as f:
    base_num = 300000
    if len(neg) > base_num * 3:
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
    else:
        neg_keep = npr.choice(len(neg), size=len(neg), replace=True)  # put back the sample
    sum_p = len(neg_keep) // 3
    pos_keep = npr.choice(len(pos), sum_p, replace=True)
    part_keep = npr.choice(len(part), sum_p, replace=True)
    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
    for item in landmark:
        f.write(item)
    print('keep neg数量：{} keep pos数量：{} keep part数量:{}, landmark数量:{}, 基数:{}'.format(len(neg_keep), len(pos_keep), len(part_keep), len(landmark), base_num))
