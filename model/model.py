import torch
import torch.nn as nn
import os
from base import BaseModel

try:
    from utils import ROOT_DIR
except:
    print('testing: model.py')

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
    def __init__(self, in_channels, out_classes=4, use_resnet=True):
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
        return x.float()                # sigmoid [0 1]: return a probabilty of each class

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
    def load_1st_model(self, load_ver, freeze_layer):
        # load checkpoint
        try:
            root_dir = ROOT_DIR
        except:
            root_dir = os.path.join(os.path.dirname(__file__), '..')

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
        if freeze_layer:
            for param in model.parameters():
                param.requires_grad = False
        
        return model, in_channels, out_classes, use_resnet

    def __init__(self, up_sample_mode='conv_transpose', load_ver='0530_225114', freeze_layer=True):
        super().__init__()
        # load 1st model with trained param
        self.model, _, self.out_classes, self.use_resnet = self.load_1st_model(load_ver, freeze_layer)
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
        self.up_conv4 = UpBlock(512, 256, self.use_resnet, up_sample_mode)
        self.up_conv3 = UpBlock(256, 128, self.use_resnet, up_sample_mode)
        self.up_conv2 = UpBlock(128, 64, self.use_resnet, up_sample_mode)
        self.up_conv1 = UpBlock(64, 32, self.use_resnet, up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(32, self.out_classes, kernel_size=1)
        # We define a BCEWithLogitsLoss since we're comparing pixel by pixel. In addition, we didn't include a final sigmoid activation as this loss function includes a sigmoid for us.
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    model = UNet(up_sample_mode='conv_transpose', load_ver='0530_225114')
    print(model)