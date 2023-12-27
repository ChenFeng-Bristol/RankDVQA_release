import os
from networks.lpips_3d import *
import torchvision.transforms as transforms
from utils import *
import numpy as np
import argparse
import json
import time
from scipy.io import savemat
import re


def calc_FR_RankDVQA(width, height, dist_path, ref_path, bitDepth):
    patch_size = 256
    num_frames_per_patch  = 12

    gt_file = open(ref_path, 'rb')
    video_file = open(dist_path, 'rb')

    if bitDepth == 8:
        bytes_per_frame = width * height + width * height * 0.5
    elif bitDepth == 10:
        bytes_per_frame = width * height * 2 + width * height
    else:
        raise Exception("Error reading file: bit depth not allowed (8 or 10)")

    frame_count = os.path.getsize(ref_path) // bytes_per_frame
    assert os.path.getsize(dist_path) // bytes_per_frame == frame_count


    all_scores = []
    for start_frame in range(0, (int(frame_count)-num_frames_per_patch), num_frames_per_patch):
        video_patches = []
        gt_patches = []
        for _ in range(num_frames_per_patch):
            video_frame = read_yuv_frame(video_file, width, height, bitDepth)
            gt_frame = read_yuv_frame(gt_file, width, height, bitDepth)

            video_frame_tensor = transform(video_frame).unsqueeze(0)
            gt_frame_tensor = transform(gt_frame).unsqueeze(0)

            video_patches.append(video_frame_tensor)
            gt_patches.append(gt_frame_tensor)

        video_patches = torch.cat(video_patches, dim=0).unsqueeze(0)
        gt_patches = torch.cat(gt_patches, dim=0).unsqueeze(0)

        with open("scores.txt", "a") as file:
            # score = moduleNetwork(gt_patches.cuda(), video_patches.cuda())
            # score = score.data.cpu().numpy().flatten() /10
            # file.write(f"{score}\n")
            # all_scores.append(score)
            for i in range(0, height-patch_size, patch_size):
                for j in range(0, width-patch_size, patch_size):
                    video_patch = video_patches[:,:,:, i:i+patch_size, j:j+patch_size]
                    gt_patch = gt_patches[:,:,:, i:i+patch_size, j:j+patch_size]
                    assert video_patch.shape[3] == patch_size and video_patch.shape[4] == patch_size, f"Invalid patch shape at ({i},{j}): {video_patch.shape}"
                    score = moduleNetwork(gt_patch.cuda(), video_patch.cuda())
                    score = score.data.cpu().numpy().flatten() /10
                    file.write(f"({i}, {j}): {score}\n")
                all_scores.append(score)

    video_file.close()
    gt_file.close()

    mean_score = np.mean(all_scores)
    print(mean_score)
    return mean_score

parser = argparse.ArgumentParser(description='')
parser.add_argument('--metric', type=str, default='FR_RankDVQA', help='evaluated metric')
parser.add_argument('--database', type=str, default='', help='root dir for datasets')
parser.add_argument('--STD', type=int, default=None, help='STD of the datasets')
parser.add_argument('--width', type=int, default='1920', help='width of the input video')
parser.add_argument('--height', type=int, default='1080', help='height of the input video')
parser.add_argument('--bitDepth', type=int, default='8', help='bitdepth of the input video')

if __name__ == '__main__':
    
    args = parser.parse_args()
    metric = args.metric
    database = args.database
    width =  args.width
    height = args.height
    bitDepth = args.bitDepth
    STD = args.STD
    
    moduleNetwork = LPIPS_3D_Diff(net='multiscale_v33').cuda()
    checkpoint = torch.load('./models/' + 'FR_model')
    moduleNetwork.load_state_dict(checkpoint['model_state_dict'])

    torch.set_grad_enabled(False)
    moduleNetwork.eval()
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
    transform = transforms.Compose(transform_list)

    metrics = {
    'FR_RankDVQA': calc_FR_RankDVQA,
    # Add other metrics here if needed
    }

    metric_name = args.metric
    metric_func = metrics.get(metric_name)
    video_dir = args.database
    database = re.split('/', args.database)[-1]

    with open(args.database + '/subj_score.json', "r") as f:
        data = json.load(f)
    data = data['test']
    ref = data['ref']
    dis = data['dis']
    mos = data['mos']
    frame_height = data['height']
    frame_width = data['width']
    framerate = data['fps']

    res_dict = {}
    labels = []
    preds = []
    
    for idx, seq in enumerate(dis):
        ref_path = os.path.join(video_dir + '/ORIG', ref[idx])
        dist_path = os.path.join(video_dir+ '/TEST', seq)
        
        t0 = time.time()
        print(metric_func)
        if metric_func is not None:
            res_dict[seq] = np.float64(metric_func(width, height, dist_path, ref_path, bitDepth))
        else:
            raise ValueError(f"No metric function found for {metric_name}")

        preds.append(res_dict[seq])
        labels.append(np.float64(mos[idx]))
        print(f'{args.metric} value for {dist_path} = {res_dict[seq]}, time taken: {time.time()-t0}')


    SROCC,PLCC,RMSE,OR = compute_metrics(preds,labels,args.STD)

    print("{phase} {metric}:\t PLCC: {plcc:.4f}\t OR: {OR:.4f}\t RMSE: {rmse:.4f}\t SROCC: {srocc:.4f}".format(phase=database,metric=args.metric, plcc=PLCC, OR=OR, rmse=RMSE, srocc=SROCC))