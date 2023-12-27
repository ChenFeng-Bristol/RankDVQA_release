import tqdm
import torch
import networks
import loss
import random
import numpy as np
import data.dataset
from utils import *
from option import opt


# fix sed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    mkdirifnotexist(opt.expdir)
    # object setting
    if opt.model == 'multiscale_v33':
        objectSettings = {
            'intEpochs': 60,
            'intBatchsize': 8,
        }
        ckptpath = None
        moduleNetwork = networks.get_model('multiscale_v33', depth_ksize=opt.depth_ksize)
        moduleLoss = loss.RealBCELoss().cuda()
        objectOptimizer = torch.optim.AdamW(list(moduleNetwork.parameters()) + list(moduleLoss.parameters()), lr=opt.lr, betas=(0.5, 0.999))
        objectScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=objectOptimizer, milestones=[30, 40], gamma=0.5)

        def forward(tenRef, tenRef2, tenFirst, tenSecond, tenJudge):
            candidate_size = range(192, 256)
            video_size = random.choice(candidate_size)

            N, B, C, H, W = tenRef.size()
            tenRef = tenRef.view(-1, C, H, W).clone()
            tenRef = torch.nn.functional.interpolate(tenRef, size=video_size, mode='bilinear', align_corners=False)
            tenRef = tenRef.view(N, B, C, video_size, video_size).clone()

            N, B, C, H, W = tenRef2.size()
            tenRef2 = tenRef2.view(-1, C, H, W).clone()
            tenRef2 = torch.nn.functional.interpolate(tenRef2, size=video_size, mode='bilinear', align_corners=False)
            tenRef2 = tenRef2.view(N, B, C, video_size, video_size).clone()

            tenFirst = tenFirst.view(-1, C, H, W).clone()
            tenFirst = torch.nn.functional.interpolate(tenFirst, size=video_size, mode='bilinear', align_corners=False)
            tenFirst = tenFirst.view(N, B, C, video_size, video_size).clone()

            tenSecond = tenSecond.view(-1, C, H, W).clone()
            tenSecond = torch.nn.functional.interpolate(tenSecond, size=video_size, mode='bilinear', align_corners=False)
            tenSecond = tenSecond.view(N, B, C, video_size, video_size).clone()

            tenDisFirst = moduleNetwork(tenRef, tenFirst)
            tenDisSecond = moduleNetwork(tenRef2, tenSecond)
            tenLoss = torch.mean(moduleLoss(tenDisFirst, tenDisSecond, tenJudge))
            return tenLoss

        def step():
            torch.nn.utils.clip_grad_norm_(moduleNetwork.parameters(), 0.1)

    else:
        raise NotImplementedError

    # 70% duo source label + 30% single source label for training
    objectTrain = torch.utils.data.DataLoader(
        batch_size=objectSettings['intBatchsize'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        dataset=data.dataset.DatasetTrain(
            dataroot=get_dataset_dir(),
            duodata=True,
        )
    )

    # label for validation
    objectTest = torch.utils.data.DataLoader(
        batch_size=objectSettings['intBatchsize'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        dataset=data.dataset.DatasetTest(
            dataroot=get_dataset_dir(),
        )
    )

    for intEpoch in range(objectSettings['intEpochs']):
        torch.set_grad_enabled(True)
        moduleNetwork.train()
        moduleLoss.train()

        for tenRef, tenRef2, tenFirst, tenSecond, tenJudge in tqdm.tqdm(objectTrain):
            objectOptimizer.zero_grad()
            forward(tenRef.cuda(), tenRef2.cuda(), tenFirst.cuda(), tenSecond.cuda(), tenJudge.cuda()).backward()
            step()
            objectOptimizer.step()

        objectScheduler.step()
        torch.set_grad_enabled(False)
        moduleNetwork.eval()
        moduleLoss.eval()
        dblTest = []
        for tenRef, tenRef2, tenFirst, tenSecond, tenJudge in tqdm.tqdm(objectTest):
            dblTest.append(forward(tenRef.cuda(), tenRef2.cuda(), tenFirst.cuda(), tenSecond.cuda(), tenJudge.cuda()).item())

        print('test ', intEpoch, ': ', np.mean(dblTest))
        with open(opt.expdir + 'train.log', 'a+') as objectFile:
            objectFile.write(str(np.mean(dblTest)) + '\n')

        _checkpoint = '{pt}_{epoch}'.format(pt=opt.expdir, epoch=intEpoch)
        torch.save({'epoch': intEpoch, 'model_state_dict': moduleNetwork.state_dict(), 'optimizer_state_dict': objectOptimizer.state_dict()}, _checkpoint)