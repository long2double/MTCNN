# pnet、rnet和onet迭代次数
end_epoch = [30, 30, 30]
# 经过多少次batch显示数据
display = 100
# 初始学习率
lr = 0.001
# 训练patch
batch_size = 384
# 学习率减少的迭代次数
lr_epoch = [6, 14, 20]
# 最小人脸大小
min_face = 40
# gpu选择
gpu = 3
# 生成hard_example的batch
batches = [2048, 256, 16]
# pnet对图像缩小倍数
stride = 2
# pnet,rnet,onet得分阈值
thresh = [0.6, 0.7, 0.7]
# 只把70%数据用作参数更新
num_keep_radio = 0.7
# 是否进行refine
restore = False
# 最后测试选择的网络

