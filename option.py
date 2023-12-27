import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='perceptual_video')
parser.add_argument('--model', type=str, default='multiscale_v33')
parser.add_argument('--expdir', type=str, default='./model/', help='exp dir')
parser.add_argument('--depth_ksize', type=int, default=1, help='depth kernel size')
parser.add_argument('--lr', type=float, default=0.0001, help='depth kernel size')
parser.add_argument('--flow', type=str2bool, default=False, help='model use flow or not')
parser.add_argument('--autodata', type=str2bool, default=True, help='model use autodata or not')
parser.add_argument('--testset', type=str, default='vmafplus', help='test set')
parser.add_argument('--norm', type=str, default='sigmoid', help='normalization function')
parser.add_argument('--window_size', type=int, default=2, help='window size')
parser.add_argument('--batchsize', type=int, default=8, help='batchsize')

opt = parser.parse_args()
