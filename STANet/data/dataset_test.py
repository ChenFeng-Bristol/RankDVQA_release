import torch
from torch.utils.data import Dataset
import numpy as np
import json
import cv2
import os
import torchvision.transforms as transforms
from networks.lpips_3d import *
from networks.multi_scale import Extractor
from networks.common import ScalingLayer
import torch.nn.functional as F

class VideoQualityDatasettest(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)['test']
        self.yuv_paths = self.data['dis']
        self.ref_yuv = self.data['ref']
        self.mos_scores = self.data['mos']
        self.width = self.data['width'][0] 
        self.height = self.data['height'][0] 
        self.num_frames = 12 
        self.dis_paths = os.path.dirname(json_path) + '/TEST'
        self.ref_paths = os.path.dirname(json_path) + '/ORIG'
        self.moduleNetwork = LPIPS_3D_Diff(net='multiscale_v33').cuda()
        loaded_state_dict = torch.load('./exp/' + 'FR_model')
        self.moduleNetwork.load_state_dict(loaded_state_dict['model_state_dict'])
        torch.set_grad_enabled(False)
        self.moduleNetwork.eval()
        self.extractor = Extractor().cuda()
        pretrained_state_dict = torch.load('./exp/FR_model')
        extractor_state_dict = {k.replace('net.moduleExtractor.', ''): v for k, v in pretrained_state_dict.items() if 'net.moduleExtractor' in k}
        self.extractor.load_state_dict(extractor_state_dict, strict=False)
        self.scaling_layer = ScalingLayer()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])


    def __len__(self):
        return len(self.yuv_paths)


    def __getitem__(self, idx):
        path_dis_1 = self.dis_paths
        yuv_path_dis_1 = self.yuv_paths[idx]
        path_ref_1 = self.ref_paths
        yuv_path_ref_1 = self.ref_yuv[idx]

        ref_path_1 = os.path.join(path_ref_1, yuv_path_ref_1)
        dis_path_1 = os.path.join(path_dis_1, yuv_path_dis_1)

        mos1 = self.mos_scores[idx]
        patch_quality_score1, Feature_layer_combine1 = self.metric_func(dis_path_1, ref_path_1, self.moduleNetwork, 8)

        return Feature_layer_combine1, mos1, patch_quality_score1

    
    def metric_func(self, dist_path, ref_path, moduleNetwork, bitDepth):
        global dis
        width = 1920
        height = 1080
        patch_size = 256
        num_frames_per_patch  = 12

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        transform = transforms.Compose(transform_list)

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

        scores = []
        Feature_layer_combine = []

        for start_frame in range(0, (int(frame_count)-num_frames_per_patch+1), num_frames_per_patch):
            video_patches = []
            gt_patches = []
            for _ in range(num_frames_per_patch):
                video_frame = read_yuv_frame(video_file, width, height, bitDepth)
                gt_frame = read_yuv_frame(gt_file, width, height, bitDepth)

                video_frame_tensor = transform(video_frame).unsqueeze(0)
                gt_frame_tensor = transform(gt_frame).unsqueeze(0)

                video_patches.append(video_frame_tensor)
                gt_patches.append(gt_frame_tensor)

            video = torch.cat(video_patches, dim=0).unsqueeze(0).cuda()
            gt = torch.cat(gt_patches, dim=0).unsqueeze(0).cuda()

            for i in range(0, video.shape[4] - 255, 110):
                for j in range(0, video.shape[3] - 255, 103):
                    video_patch = video[:, :, :, j:j + 256, i:i + 256]
                    gt_patch = gt[:, :, :, j:j + 256, i:i + 256]
                    in0_input = self.scaling_layer(video_patch)
                    in0_input = in0_input.cuda()
                    B, V, C, H, W = in0_input.size()
                    in0_input = in0_input.view(B * V, C, H, W)

                    dis = moduleNetwork(gt_patch, video_patch)
                    Featuremap = self.extractor(in0_input)

                    downsampled_tensor1 = F.interpolate(Featuremap[2], size=(4, 4), mode='bilinear', align_corners=True)

                    Feature_layer_combine.append(torch.cat((downsampled_tensor1, Featuremap[5]), dim=1))
                    dis = dis.data.cpu().numpy().flatten() /10
                    scores.append(dis)
        video_file.close()
        gt_file.close()

        return scores, Feature_layer_combine

def read_yuv_frame(fid, width, height, bitDepth):

    if bitDepth == 8:
        dtype = np.uint8
    elif bitDepth == 10:
        dtype = np.uint16
    else:
        raise Exception("Error reading file: bit depth not allowed (8 or 10 )")

    y_size = width * height
    y_data = np.frombuffer(fid.read(y_size * dtype().itemsize), dtype).reshape((height, width))
    

    uv_size = y_size // 2  
    fid.read(uv_size * dtype().itemsize)
    
    if bitDepth == 10:
        y_data = y_data / 1023.0
    if bitDepth == 8:
        y_data = y_data / 255.0
    
    return y_data.astype(np.float32)
