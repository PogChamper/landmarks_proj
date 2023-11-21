from utils import *
from read_newpoint_data import *
from sklearn.model_selection import train_test_split
from PointsDataset import PointsDataset
import glob
import os
from onet import ONet
from onet_ch import ONetCh
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
import argparse
import random


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
    train_dataset = PointsDataset(df_train,(48,48))
    val_dataset = PointsDataset(df_test,(48,48))
    return train_dataset,val_dataset


def prepare_data_test(dataset):
    paths1 = glob.glob(f'data/landmarks_task_rects/{dataset}/test/*.jpg')
    paths2 = glob.glob(f'data/landmarks_task_rects/{dataset}/test/*.png')
    paths = paths1 + paths2
    df_test = read_newpointdata(paths)
    test_dataset = PointsDataset(df_test,(48,48))
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
               num_workers=4, use_cuda=True, model_store_path='Onet_train'):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)
            
    loss_train_history = []
    loss_val_history = []

    losses_train = AverageMeter()
    losses_val = AverageMeter()
    
    net = ONet(is_train=True)
    
    net.train()
    if use_cuda:
        net.cuda()
    
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_sceduler)
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

                loss = criterion(output, gt_landmark)

                losses_val.update(loss.item(), image.size(0))
                
        loss_train_history.append(losses_train.avg)
        loss_val_history.append(losses_val.avg)
        
        lr_scheduler.step()

        print("Epoch: %d, landmark loss train: %s, landmark loss val: %s " % (cur_epoch, loss_train_history[-1],loss_val_history[-1]))
        if loss_val_history[-1]<best_loss:
            best_loss = loss_val_history[-1]
            torch.save(net.state_dict(), f"{model_store_path}/best_epoch.pt")
        # torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    return loss_train_history, loss_val_history


def test(model_path, lst_paths, test_dataset, res_dir, batch_size=64, num_workers=8, use_cuda=True):
    
    make_all_dirs(res_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    # CHANGE MODEL FOR TEST HERE:

    net = ONet(is_train=True)
    #net = ONetCh(is_train=True)

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
    parser.add_argument('--num_workers', action='store', type=int, help='',
                        default=4)
    parser.add_argument('--max_epoch', action='store', type=int, help='',
                        default=50)
    parser.add_argument('--result_dir', action='store', type=str, help='',
                        default='result_onet')
    parser.add_argument('--base_lr', action='store', type=str, help='',
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
                          num_workers=args.num_workers,use_cuda=args.cuda, model_store_path=args.output_path)
        
    if args.mode=='test':
        test_dataset,paths  = prepare_data_test(args.dataset)
        print(len(test_dataset))
        test(f'{args.output_path}/best_epoch.pt', paths, test_dataset, args.result_dir, batch_size=args.batch_size, num_workers=args.num_workers,use_cuda=args.cuda)        

if __name__ == '__main__':
    main()