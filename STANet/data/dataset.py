import torch
from torch.utils.data import Dataset
import numpy as np
import json
import cv2
import os
import torchvision.transforms as transforms
from networks.lpips_3d import *
from PIL import Image
from networks.multi_scale import Extractor
from networks.common import ScalingLayer
import torch.nn.functional as F

class VideoQualityDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)['train']

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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.yuv_paths)


    def __getitem__(self, idx):
        idx2 = np.random.choice([i for i in range(len(self.yuv_paths)) if i != idx])
        path_dis_1 = self.dis_paths
        yuv_path_dis_1 = self.yuv_paths[idx]
        path_ref_1 = self.ref_paths
        yuv_path_ref_1 = self.ref_yuv[idx]

        path_dis_2 = self.dis_paths
        yuv_path_dis_2 = self.yuv_paths[idx2]
        path_ref_2 = self.ref_paths
        yuv_path_ref_2 = self.ref_yuv[idx2]

        ref_path_1 = os.path.join(path_ref_1, yuv_path_ref_1)
        ref_path_2 = os.path.join(path_ref_2, yuv_path_ref_2)

        dis_path_1 = os.path.join(path_dis_1, yuv_path_dis_1)
        dis_path_2 = os.path.join(path_dis_2, yuv_path_dis_2)

        patch_quality_score1, Feature_layer_combine1 = self.metric_func(dis_path_1, ref_path_1, self.moduleNetwork, 8)
        patch_quality_score2, Feature_layer_combine2 = self.metric_func(dis_path_2, ref_path_2, self.moduleNetwork, 8)

        mos1 = self.mos_scores[idx]
        mos2 = self.mos_scores[idx2]

        return mos1, mos2, patch_quality_score1, patch_quality_score2,Feature_layer_combine1,Feature_layer_combine2

    def metric_func(self, dist_path, ref_path, moduleNetwork, bitDepth):
        global dis
        width = 1920
        height = 1080
        if bitDepth == 8:
            multiplier = 1
            elementType = 'yuv420p'
        elif bitDepth == 10:
            multiplier = 2
            elementType = 'yuv420p10le'
        else:
            raise Exception("Error reading file: bit depth not allowed (8 or 10 )")
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)

        os.system(
            'ffmpeg -f rawvideo -vcodec rawvideo -s {}x{} -r 25 -pix_fmt {} -i {} -c:v libx264 -preset ultrafast -qp 0 vfips_dist_1.mp4'.format(
                width, height, elementType, dist_path))

        dstpath = 'vfips_dist_1.mp4'
        cap_video = cv2.VideoCapture(dstpath)
        if not cap_video.isOpened():
            print("Error opening Dis video")

        os.system(
            'ffmpeg -f rawvideo -vcodec rawvideo -s {}x{} -r 25 -pix_fmt {} -i {} -c:v libx264 -preset ultrafast -qp 0 vfips_ref_1.mp4'.format(
                width, height, elementType, ref_path))
        gtpath = 'vfips_ref_1.mp4'
        cap_gt = cv2.VideoCapture(gtpath)
        if not cap_gt.isOpened():
            print("Error opening GT video")

        assert int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) == int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))


        scores = []
        Feature_layer_combine = []

        for start_id in range(0, 121-12, 12):
            video = []
            gt = []
            for i in range(12):
                ret, videoimg = cap_video.read()
                assert ret
                videoimg = cv2.cvtColor(videoimg, cv2.COLOR_BGR2RGB)
                videoimg = Image.fromarray(videoimg)
                videoimg = transform(videoimg).unsqueeze(0)

                ret, gtimg = cap_gt.read()
                assert ret
                gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
                gtimg = Image.fromarray(gtimg)
                gtimg = transform(gtimg).unsqueeze(0)

                video.append(videoimg)
                gt.append(gtimg)

            video = torch.cat(video, dim=0)
            gt = torch.cat(gt, dim=0)

            video = video.unsqueeze(0).cuda()
            gt = gt.unsqueeze(0).cuda()

            for i in range(0, video.shape[4] - 255, 110):
                for j in range(0, video.shape[3] - 255, 103):
                    video_patch = video[:, :, :, j:j + 256, i:i + 256]
                    gt_patch = gt[:, :, :, j:j + 256, i:i + 256]

                    video_patch = video_patch.to('cuda')
                    gt_patch = gt_patch.to('cuda')
                    self.scaling_layer.shift = self.scaling_layer.shift.to('cuda')
                    self.scaling_layer.scale = self.scaling_layer.scale.to('cuda')

                    # Now apply the scaling layer
                    in0_input = self.scaling_layer(video_patch)
                    in0_input = in0_input.cuda()
                    B, V, C, H, W = in0_input.size()
                    in0_input = in0_input.view(B * V, C, H, W)

                    dis = moduleNetwork(gt_patch, video_patch)
                    Featuremap = self.extractor(in0_input)
                    
                    downsampled_tensor1 = F.interpolate(Featuremap[2], size=(4, 4), mode='bilinear', align_corners=True)

                    Feature_layer_combine.append(torch.cat((downsampled_tensor1, Featuremap[5]), dim=1))

                    dis = dis.data.cpu().numpy().flatten()/10
                    scores.append(dis)

        cap_video.release()
        cap_gt.release()

        # delete files, modify command accordingly
        os.remove('vfips_dist_1.mp4')
        os.remove('vfips_ref_1.mp4')
        return scores, Feature_layer_combine