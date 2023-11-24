from utils import *
from read_newpoint_data import *
from sklearn.model_selection import train_test_split
from PointsDataset import PointsDataset
import glob
import os
from onet import ONet
from onet_ch import ONetCh
from onetResidal import ONetRes
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
import argparse
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from torch import nn
import math
import pytorch_warmup as warmup

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, theta=0.5, epsilon=1.0,\
               whetherWeighted=False, dilaStru=3, w=10, device="cuda"):
        super(AdaptiveWingLoss, self).__init__()
        self.device = device
        self.alpha = torch.Tensor([alpha]).to(device)
        self.omega = torch.Tensor([omega]).to(device)
        self.theta = torch.Tensor([theta]).to(device)
        self.epsilon = torch.Tensor([epsilon]).to(device)
        self.dilationStru = dilaStru
        self.w = torch.Tensor([w]).to(device)
        self.tmp = torch.Tensor([self.theta / self.epsilon]).to(device)
        self.wetherWeighted = whetherWeighted

    '''
    #param predictions: predicted heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
    #param targets: ground truth heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
    '''
    def forward(self, predictions, targets):
        deltaY = predictions - targets
        deltaY = torch.abs(deltaY)
        alphaMinusY = self.alpha - targets
        a = self.omega / self.epsilon * alphaMinusY / (1 + self.tmp.pow(alphaMinusY))\
            * self.tmp.pow(alphaMinusY - 1)
        c = self.theta * a - self.omega * torch.log(1 + self.tmp.pow(alphaMinusY))

        l = torch.where(deltaY < self.theta,
                        self.omega * torch.log(1 + (deltaY / self.epsilon).pow(alphaMinusY)),
                        a * deltaY - c)
        if self.wetherWeighted:
            weightMap = self.grayDilation(targets, self.dilationStru)
            weightMap = torch.where(weightMap >= 0.2, torch.Tensor([1]).to(self.device),\
                                    torch.Tensor([0]).to(self.device))
            l = l * (self.w * weightMap + 1)

        l = torch.mean(l)

        return l
    
    def grayDilation(self, heatmapGt, structureSize):
        batchSize, landmarkNum, heatmapSize, _ = heatmapGt.shape
        weightMap = heatmapGt.clone()
        step = structureSize // 2
        for i in range(1, heatmapSize-1, 1):
            for j in range(1, heatmapSize-1, 1):
                weightMap[:, :, i, j] = torch.max(heatmapGt[:, :, i - step: i + step + 1,\
                                        j - step: j + step + 1].contiguous().view(batchSize,\
                                        landmarkNum, structureSize * structureSize), dim=2)[0]

        return weightMap  
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        #print(loss1, loss2)
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
    

SIZE = 128


train_transform = A.Compose(
    [
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.01),
        A.RandomBrightnessContrast(p=0.05),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.03),
        A.AdvancedBlur(p=0.1),
        A.ISONoise(p=0.1, intensity=(0.10000000149011612, 0.5), color_shift=(0.009999999776482582, 0.25)),
        A.MultiplicativeNoise(always_apply=False, p=0.3, multiplier=(0.9, 1.1), per_channel=False, elementwise=False), 
        A.ChannelDropout(p=0.05, channel_drop_range=(1, 1), fill_value=0),
        A.GaussNoise(p=0.1, var_limit=(10.0, 500.0)),
        #A.Rotate(p=0.8, limit=(-15, 15), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
        #A.RandomCrop(height=10, width=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(SIZE, SIZE),
        #transforms.ToTensor(),
        ToTensorV2(),
    ]
)


val_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(SIZE, SIZE),
        #transforms.ToTensor(),
        ToTensorV2(),
    ]
)


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prepare_data_train():
    paths1 = glob.glob('./data/landmarks_task_rects/*/train/*.jpg')
    paths2 = glob.glob('./data/landmarks_task_rects/*/train/*.png')
    paths = paths1 + paths2
    df_prepared = read_newpointdata(paths)
    df_train, df_test = train_test_split(df_prepared,test_size=0.2,random_state=20)
    train_dataset = PointsDataset(df_train,(64,64), transform=train_transform)
    val_dataset = PointsDataset(df_test,(64,64), transform=val_transform)
    return train_dataset,val_dataset


def prepare_data_test(dataset):
    paths1 = glob.glob(f'data/landmarks_task_rects/{dataset}/test/*.jpg')
    paths2 = glob.glob(f'data/landmarks_task_rects/{dataset}/test/*.png')
    paths = paths1 + paths2
    df_test = read_newpointdata(paths)
    test_dataset = PointsDataset(df_test,(64,64), transform=val_transform)
    return test_dataset, df_test.iloc[:, 0].values


def make_all_dirs(res_dir):
    datasets = ['300W','Menpo']
    modes = ['train','test']
    for dataset in datasets:
        for mode in modes:
            if not os.path.exists(f'data/{res_dir}/{dataset}/{mode}'):
                os.makedirs(f'data/{res_dir}/{dataset}/{mode}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_onet(train_dataset, val_dataset, end_epoch,
               batch_size, base_lr=0.01, step_sceduler=25,
               num_workers=4, use_cuda=True, model_store_path='Onet_train', model='ONet'):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)
            
    loss_train_history = []
    loss_val_history = []

    losses_train = AverageMeter()
    losses_val = AverageMeter()
    
    if model == 'ONet':
        net = ONet(is_train=True)
    elif model == 'ONetRes':
        net = ONetRes(is_train=True)
    else:
        net = ONetCh(is_train=True)

    net.train()

    if use_cuda:
        net.cuda()
    #criterion = torch.nn.MSELoss(size_average=True).cuda()
    #criterion = WingLoss().cuda()
    criterion = AdaptiveWingLoss().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=5e-3)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9) #lr=base_lr)
    #optimizer = NoamOpt(input_opts['d_model'], 500,
    #        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    best_loss = 100000000000

    for cur_epoch in range(end_epoch):
        net.train()
        for batch_idx, (image,gt_landmark, _) in enumerate(tqdm(train_loader)):

            if use_cuda:
                image = image.cuda()
                gt_landmark = gt_landmark.cuda()

            landmark_offset_pred = net(image)
            loss = criterion(gt_landmark,landmark_offset_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses_train.update(loss.item(), image.size(0))

        net.eval()
        with torch.no_grad():
            for batch_idx,(image,gt_landmark,_) in enumerate(val_loader):
                
                if use_cuda:
                    image = image.cuda()
                    gt_landmark = gt_landmark.cuda()
                    
                output = net(image)

                loss = criterion(gt_landmark, output)

                losses_val.update(loss.item(), image.size(0))
                
        loss_train_history.append(losses_train.avg)
        loss_val_history.append(losses_val.avg)
        
        with warmup_scheduler.dampening():
            scheduler.step(loss_val_history[-1])
        #scheduler.step(loss_val_history[-1])
        my_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'LR is: {my_lr}')

        print("Epoch: %d, landmark loss train: %s, landmark loss val: %s " % (cur_epoch, loss_train_history[-1],loss_val_history[-1]))
        if loss_val_history[-1]<best_loss:
            best_loss = loss_val_history[-1]
            torch.save(net.state_dict(), f"{model_store_path}/best_epoch.pt")
        # torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    return loss_train_history, loss_val_history


def test(model_path, lst_paths, test_dataset, res_dir, batch_size=64, num_workers=8, use_cuda=True, model='ONet'):
    
    make_all_dirs(res_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    # CHANGE MODEL FOR TEST HERE:
    if model == 'ONet':
        net = ONet(is_train=True)
    elif model == 'ONetRes':
        net = ONetRes(is_train=True)
    else:
        net = ONetCh(is_train=True)

    net.load_state_dict(torch.load(model_path))
    net.eval()
    if use_cuda:
        net.cuda()
    with torch.no_grad():
        for batch_idx,(image,gt_landmark,meta) in enumerate(tqdm(test_loader)):

            if use_cuda:
                image = image.cuda()
                # gt_landmark = gt_landmark.cuda()

            output = net(image)

            score_map = output.data.cpu()

            for n in range(score_map.size(0)):
                real_x = score_map[n, ::2].numpy() * meta['new_size'][0][n].numpy() + meta['left_corner_xy'][0][n].numpy()
                real_y = score_map[n, 1::2].numpy() * meta['new_size'][1][n].numpy() + meta['left_corner_xy'][1][n].numpy()
                real_xy = np.hstack((real_x.reshape(-1, 1), real_y.reshape(-1, 1))) 
                path = lst_paths[meta['index'][n].item()]
                with open(path.replace(str(Path(path).suffix),'.pts').replace('landmarks_task_rects',res_dir),'w') as f:
                    f.write('\n'.join('version: 1\nn_points:  68\n{'.split('\n') + 
                        [str(x)[1:-1].replace(',', '') for x in real_xy.tolist()] + 
                        list(['}'])))
                    

def main():
    parser = argparse.ArgumentParser(description='Onet train/inf',
                                     add_help=True)
    parser.add_argument('--mode', action='store', type=str, help='', default='train')
    parser.add_argument('--output_path', action='store', type=str, help='', default='Onet-changed-train')
    parser.add_argument('--cuda', action='store_true',help='', default=True)
    parser.add_argument('--batch_size', action='store', type=int, help='',
                        default=32)
    parser.add_argument('--model', action='store', type=str, help='',
                        default='ONet')
    parser.add_argument('--num_workers', action='store', type=int, help='',
                        default=4)
    parser.add_argument('--max_epoch', action='store', type=int, help='',
                        default=50)
    parser.add_argument('--result_dir', action='store', type=str, help='',
                        default='result_onet')
    parser.add_argument('--base_lr', action='store', type=float, help='',
                        default=0.001)
    parser.add_argument('--step_scheduler', action='store', type=int, help='',
                        default=25)
    parser.add_argument('--dataset', action='store', type=str, help='',
                        default='300W')
    args = parser.parse_args()
    
    seed_everything(42)

    if args.mode=='train':
        train_dataset,val_dataset = prepare_data_train()
        _, _ = train_onet(train_dataset, val_dataset, end_epoch=args.max_epoch,
                          batch_size=args.batch_size,base_lr=args.base_lr,step_sceduler=args.step_scheduler,
                          num_workers=args.num_workers,use_cuda=args.cuda, model_store_path=args.output_path, 
                          model=args.model)
        
    if args.mode=='test':
        test_dataset,paths  = prepare_data_test(args.dataset)
        print(len(test_dataset))
        test(f'{args.output_path}/best_epoch.pt', paths, test_dataset, args.result_dir, 
             batch_size=args.batch_size, num_workers=args.num_workers,
             use_cuda=args.cuda, model=args.model)        

if __name__ == '__main__':
    main()