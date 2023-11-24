import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, (nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)

        
class ONetCh(nn.Module):

    def __init__(self,is_train=False, use_cuda=True):
        super(ONetCh, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 126    # 62 #30
            nn.BatchNorm2d(32),
            DropBlock2D(0.25, 5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 61   # 29 #13
            nn.BatchNorm2d(64),
            DropBlock2D(0.25, 5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 29    #13 #5
            nn.BatchNorm2d(128),
            DropBlock2D(0.25, 5),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), 
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13   # 5
            nn.BatchNorm2d(256),
            DropBlock2D(0.25, 5),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.SiLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2), # 13   # 5
            #nn.BatchNorm2d(512),
            #DropBlock2D(0.25, 512),
            #nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.SiLU(),
            DropBlock2D(0.25, 5),
            #nn.Dropout(0.25) 
        )
        self.conv4 = nn.Conv2d(64,128,kernel_size=2,stride=1) 
        self.conv5 = nn.Linear(512*2*2, 1024)
        self.prelu5 = nn.SiLU()  # prelu5
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(1024, 136)
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        landmark = self.conv6_3(x)
        if self.is_train is True:
            return landmark
        return landmark
