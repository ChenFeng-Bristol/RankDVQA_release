import os
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, duodata=True, booltrain=False, boolBinary=True):
        self.dataroot = dataroot
        self.duo = duodata
        self.booltrain = booltrain
        self.boolBinary = boolBinary

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        videodir = self.videolist[index]
        videodir, label = videodir.split(' ')

        video1 = []
        video2 = []
        gt = []
        gt2 = []

        v1path = '%s/LR1.yuv' % (videodir)
        v2path = '%s/LR2.yuv' % (videodir)
        
        if "Training_Content_duo" in videodir:
            gtpath = '%s/gt1.yuv' % (videodir)
            gt2path = '%s/gt2.yuv' % (videodir)
        else:
            gtpath = '%s/gt.yuv' % (videodir)
            gt2path = '%s/gt.yuv' % (videodir)
        
        for i in range(12):

            v1img = yuv_to_rgb_single_frame(v1path, 256, 256, 10, i)
            v2img = yuv_to_rgb_single_frame(v2path, 256, 256, 10, i)
            gtimg = yuv_to_rgb_single_frame(gtpath, 256, 256, 10, i)
            gt2img = yuv_to_rgb_single_frame(gt2path, 256, 256, 10, i)

            v1img = Image.fromarray(v1img.astype(np.uint8))
            v1img = self.transform(v1img).unsqueeze(0)

            v2img = Image.fromarray(v2img.astype(np.uint8))
            v2img = self.transform(v2img).unsqueeze(0)

            gtimg = Image.fromarray(gtimg.astype(np.uint8))
            gtimg = self.transform(gtimg).unsqueeze(0)

            gt2img = Image.fromarray(gt2img.astype(np.uint8))
            gt2img = self.transform(gt2img).unsqueeze(0)

            video1.append(v1img)
            video2.append(v2img)
            gt.append(gtimg)
            gt2.append(gt2img)

        video1 = torch.cat(video1, dim=0)
        video2 = torch.cat(video2, dim=0)
        gt = torch.cat(gt, dim=0)
        gt2 = torch.cat(gt2, dim=0)

        if self.boolBinary:
            label = np.float(float(label) > 0.5)
        judge_img = np.array(float(label)).reshape((1,))
        judge_img = torch.FloatTensor(judge_img)

        if self.booltrain and random.random() < 0.5:
            video1, video2 = video2, video1
            flow1, flow2 = flow2, flow1
            judge_img = 1 - judge_img


        return gt, gt2, video1, video2, judge_img

    def __len__(self):
        return len(self.videolist)


class DatasetTrain(Dataset):
    def __init__(self, dataroot, duodata=True):
        super(DatasetTrain, self).__init__(dataroot, duodata, True, boolBinary=True)

        videolist = []
        videolist_single = []
        videolist_duo = []
        with open(self.dataroot + 'VMAF_train_single.txt', 'r') as f:
            videolist_single += f.read().splitlines()
        for i in range(len(videolist_single)):
            videolist_single[i]= self.dataroot +'/Training_Content_single'+ '/' + videolist_single[i]

        if duodata:
            with open(self.dataroot + 'VMAF_train_duo.txt', 'r') as f:
                videolist_duo += f.read().splitlines()
            for i in range(len(videolist_duo)):
                videolist_duo[i]= self.dataroot +'/Training_Content_duo'+ '/' + videolist_duo[i]    
        else:
            print('=> Warning: No duo data for training')

        videolist = videolist_single + videolist_duo

        random.shuffle(videolist)
        self.videolist = videolist
        self.videolist_single = videolist_single
        self.videolist_duo = videolist_duo
        print('=> load %d samples from %s' % (len(self.videolist_single), dataroot))
        print('=> load %d samples from %s' % (len(self.videolist_duo), dataroot))
        print('=> load %d samples from %s' % (len(self.videolist), dataroot))



class DatasetTest(Dataset):
    def __init__(self, dataroot, duodata=True):
        super(DatasetTest, self).__init__(dataroot, duodata, False, boolBinary=False)
        with open(self.dataroot + 'VMAF_train_duo_Val.txt', 'r') as f:
            videolist = f.read().splitlines()
        for i in range(len(videolist)):
            videolist[i]= self.dataroot +'/Training_Content_duo_Val'+ '/' + videolist[i]    
        self.videolist = videolist
        self.videolist.sort()
        print('=> load %d samples from %s' % (len(self.videolist), dataroot))


