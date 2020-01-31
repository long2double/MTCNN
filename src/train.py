from network.model import p_net, r_net, o_net
from network.train_model import train
import argparse
import os
import sys
import config as FLAGS

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)

root_dir = os.path.dirname(__file__).split('MTCNN')[0]
project_dir = os.path.dirname(__file__).split('MTCNN')[1]


def main(save_dir, input_size):
    base_dir = os.path.join(save_dir, input_size)
    net = None
    if input_size == '12':
        net = 'pnet'
        net_factory = p_net
        end_epoch = FLAGS.end_epoch[0]
    elif input_size == '24':
        net = 'rnet'
        net_factory = r_net
        end_epoch = FLAGS.end_epoch[1]
    else:
        net = 'onet'
        net_factory = o_net
        end_epoch = FLAGS.end_epoch[2]

    model_dir = os.path.join(root_dir + 'MTCNN' + '/checkpoint', net)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_prefix = os.path.join(model_dir, net)
    display = FLAGS.display
    lr = FLAGS.lr
    train(net_factory, model_prefix, end_epoch, base_dir, display, lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--save_dir', type=str, default='/mnt/data/changshuang/gen_data',
                        help='保存图片路径')
    parser.add_argument('--input_size', type=str, required=True, choices=['12', '24', '48'],
                        help='对于具体网络输入图片的大小')
    args = parser.parse_args()
    main(args.save_dir, args.input_size)

    