import torch
import torch.nn as nn
import os
from base import BaseModel

from utils import ROOT_DIR

class DoubleConv(nn.Module):
    # conv_block: Conv => BN => ReLu => (MaxPool)
    def conv_block(self, in_channels, out_channels, pool=False):
        layers =    [nn.Conv2d(
                        in_channels,             # no. of layers of pervious layer
                        out_channels,            # n_filters
                        kernel_size=3,           # filter size (3/5)
                        stride=1,                # filter movement/step
                        padding=1                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1 (1/2)
                    ), 
                    nn.BatchNorm2d(num_features=out_channels),   # Normally, num_features = no. of layers of pervious layer
                    nn.ReLU(inplace=True)        # inplace=True to save memory
                    ]
        if pool: layers.append(nn.MaxPool2d(kernel_size=2, stride=2))   # width/2
        return nn.Sequential(*layers)
        
    def __init__(self, in_channels, out_channels, use_resnet):
        super().__init__()
        self.use_resnet = use_resnet
        self.conv1 = self.conv_block(in_channels, out_channels, pool=False)
        self.conv2 = self.conv_block(out_channels, out_channels, pool=False)
        if self.use_resnet:
            self.convRes = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_resnet:
            x = self.convRes(identity) + x         # where Resnet happens here
            x = self.relu(x)
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_resnet):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, use_resnet)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class FirstHalfUNet(BaseModel):
    def __init__(self, in_channels, out_classes, use_resnet):
        super().__init__()
        # BN first
        self.bn_input = nn.BatchNorm2d(1)
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 32, use_resnet)
        self.down_conv2 = DownBlock(32, 64, use_resnet)
        self.down_conv3 = DownBlock(64, 128, use_resnet)
        self.down_conv4 = DownBlock(128, 256, use_resnet)
        # Bottleneck
        self.double_conv = DoubleConv(256, 512, use_resnet)
        # Dense Layer
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, out_classes),
            nn.BatchNorm1d(out_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bn_input(x)            # (batch_size,  1, 256, 1600)
        x, _ = self.down_conv1(x)       # (batch_size, 32, 128, 800)
        x, _ = self.down_conv2(x)       # (batch_size, 64, 64, 400)
        x, _ = self.down_conv3(x)       # (batch_size, 128, 32, 200)
        x, _ = self.down_conv4(x)       # (batch_size, 256, 16, 100)
        x = self.double_conv(x)         # (batch_size, 512, 16, 100)
        x = self.globalavgpool(x)       # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)       # (batch_size, 512)   cannot use flatten here which will be (batch_size*256*16*16)
        x = self.fc1(x)                 # (batch_size, 128)
        x = self.fc2(x)                 # (batch_size, 4)
        return x                        # sigmoid [0 1]: return a probabilty of each class

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_resnet, up_sample_mode='conv_transpose'):
        super().__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels+out_channels, out_channels, use_resnet)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        x = self.double_conv(x)
        return x

class UNet(BaseModel):
    def load_1st_model(self, load_ver):
        # load checkpoint
        root_dir = ROOT_DIR
        checkpoint_path = os.path.join(root_dir, 'saved/models/Steel_Defect_Detection', load_ver, 'model_best.pth')
        try:
            checkpoint = torch.load(checkpoint_path)
        except:
            print('Wrong load checkpoint path')
        
        in_channels = checkpoint['config']['arch']['args']['in_channels']
        out_classes = checkpoint['config']['arch']['args']['out_classes']
        use_resnet = checkpoint['config']['arch']['args']['use_resnet']
    
        # load model architecture
        model = FirstHalfUNet(in_channels=in_channels, out_classes=out_classes, use_resnet=use_resnet)

        # load model param from FirstHalfUNet checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def __init__(self, in_channels, out_classes, use_resnet, up_sample_mode='conv_transpose', load_ver=None):
        super().__init__()
        # load 1st model with trained param
        if load_ver is None:
            self.model = FirstHalfUNet(in_channels, out_classes, use_resnet)
        else:
            self.model = self.load_1st_model(load_ver)

        # BN first
        self.bn_input = list(self.model.children())[0]
        # Downsampling Path
        self.down_conv1 = nn.Sequential(list(self.model.children())[1])
        self.down_conv2 = nn.Sequential(list(self.model.children())[2])
        self.down_conv3 = nn.Sequential(list(self.model.children())[3])
        self.down_conv4 = nn.Sequential(list(self.model.children())[4])
        # Bottleneck
        self.double_conv = nn.Sequential(list(self.model.children())[5])
        
        # Upsampling Path
        self.up_sample_mode = up_sample_mode
        self.up_conv4 = UpBlock(512, 256, use_resnet, up_sample_mode)
        self.up_conv3 = UpBlock(256, 128, use_resnet, up_sample_mode)
        self.up_conv2 = UpBlock(128, 64, use_resnet, up_sample_mode)
        self.up_conv1 = UpBlock(64, 32, use_resnet, up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(32, out_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()             # If we use BCEWithLogitsLoss which already include sigmoid, so no need sigmoid. If binary_cross_entropy, need sigmoid.

    def forward(self, x):
        x = self.bn_input(x)                    # (batch_size,  1, 256, 1600)
        x, skip1_out = self.down_conv1(x)       # (batch_size, 32, 128, 800)
        x, skip2_out = self.down_conv2(x)       # (batch_size, 64, 64, 400)
        x, skip3_out = self.down_conv3(x)       # (batch_size, 128, 32, 200)
        x, skip4_out = self.down_conv4(x)       # (batch_size, 256, 16, 100)
        x = self.double_conv(x)                 # (batch_size, 512, 16, 100)
        x = self.up_conv4(x, skip4_out)         # (batch_size, 256, 32, 200)
        x = self.up_conv3(x, skip3_out)         # (batch_size, 128, 64, 400)
        x = self.up_conv2(x, skip2_out)         # (batch_size, 64, 128, 800)
        x = self.up_conv1(x, skip1_out)         # (batch_size, 32, 256, 1600)
        x = self.conv_last(x)                   # (batch_size, 4, 256, 1600)
        mask_pred = self.sigmoid(x)
        label_pred = mask_pred.sum(-1).sum(-1)>=115     # (batch_size, 4)   # 115 is referenced from the min of mask.sum() in training set
        del x
        return mask_pred, label_pred.float()

if __name__ == '__main__':
    model = UNet(in_channels=1, out_classes=4, use_resnet=False, up_sample_mode='conv_transpose', load_ver=None)    #load_ver='0530_225114'
    #model = FirstHalfUNet(in_channels=1, out_classes=4, use_resnet=False)
    print(model)