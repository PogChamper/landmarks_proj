import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2023)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        #nn.init.constant(m.bias, 0.1)
    elif isinstance(m, (nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



# class LossFn:
#     def __init__(self):
#         # loss function
#         # self.land_factor = landmark_factor
#         self.loss_landmark = nn.MSELoss()

#     def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
#         pred_landmark = torch.squeeze(pred_landmark)
#         gt_landmark = torch.squeeze(gt_landmark)

#         return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ONetRes(nn.Module):

    def __init__(self,is_train=False, use_cuda=True):
        super(ONetRes, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.conv1 = ResidualBlock(3,32,stride=2)
        # blocks = [
        #     ResidualBlock(3,32,stride=2),  # pool1 48x48 -> 24x24
        #     ResidualBlock(32,64,stride=2),  # pool2 24x24 -> 12x12
        #     ResidualBlock(64,64,stride=2),  # pool3 12x12 -> 6x6
        #     ResidualBlock(64,128,stride=2) # pool3 6x6 -> 3x3
        # ]
        blocks = [
            ResidualBlock(3,32,stride=2),  #72 - 36 # pool1 64 - 32
            ResidualBlock(32,64,stride=2),  #36 - 18 # pool2 32 - 16
            ResidualBlock(64,64,stride=2),  #18 - 9 # pool3 16 - 8
            ResidualBlock(64,128,stride=2), #9 - 5 # pool3 8 - 4
            ResidualBlock(128,256,stride=2), #5 - 3 # 4 - 2
            ResidualBlock(256,512,stride=2) # 3 - 2
        ]

        self.pre_layer = nn.Sequential(*blocks)
        self.conv4 = nn.Conv2d(512,1024,kernel_size=1,stride=1) # 2 - 2
        self.prelu4 = nn.PReLU() 
        self.conv5 = nn.Linear(1024*2*2, 512)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(512, 136)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        #det = torch.sigmoid(self.conv6_1(x))
        #box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        if self.is_train is True:
            return landmark
        #landmard = self.conv5_3(x)
        return landmark

