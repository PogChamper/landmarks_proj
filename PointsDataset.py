import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
#import torchvision.transforms as transforms
import cv2
import torch

torch.manual_seed(2023)
np.random.seed(2023)

class PointsDataset(Dataset):
    def __init__(self, df, imgsz, transform):
        self.df = df
        self.input_size = imgsz
        # self.img_dir = img_dir
        #self.transform = transforms.ToTensor()
        self.transform = transform
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = cv2.imread(img_path)
        w, h, x1,y1 = self.df.iloc[idx].values[-4:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #labels = self.df.iloc[idx].values[1:-4].astype(float).reshape(-1, 2).tolist()
        
        labels = self.df.iloc[idx].values[1:-4].astype(float)
        #keypoints = labels.reshape(-1, 2).tolist()
        #labels = torch.Tensor(labels)
        #print(keypoints)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            #keypoints = transformed['keypoints']
            #print(labels[0])
        #image = image.astype(np.float32)
        #image = (image/255.0 - self.mean) / self.std
        #image = cv2.resize(image, self.input_size, interpolation = cv2.INTER_AREA)
        #image = self.transform(image=image)

        #print(labels.shape)
        
        meta = {'index': idx, 'new_size': (w, h), 'left_corner_xy': (x1, y1)}
        labels = torch.Tensor(labels)
        #print(labels.shape)
        return image, labels, meta
# def main():
#     df_train = pd.read_csv('/home/baishev/projects/landmarks/df.csv')
#     train_dataset = PointsDataset(df_train,(48,48))
#     print(train_dataset[0])

# if __name__ == '__main__':
#     main()