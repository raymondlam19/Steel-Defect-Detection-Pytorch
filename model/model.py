import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# conv_block: Conv => BN => ReLu => (MaxPool)
def conv_block(in_channels, out_channels, pool=False):
    layers =    [nn.Conv2d(
                    in_channels,             # no. of layers of pervious layer
                    out_channels,            # n_filters
                    kernel_size=5,           # filter size (3/5)
                    stride=1,                # filter movement/step
                    padding=2                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1 (1/2)
                ), 
                nn.BatchNorm2d(num_features=out_channels),   # Normally, num_features = no. of layers of pervious layer
                nn.ReLU(inplace=True)        # inplace=True to save memory
                ]
    if pool: layers.append(nn.MaxPool2d(kernel_size=2, stride=2))   # width/2
    return nn.Sequential(*layers)

class CNN(BaseModel):        
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)   # BN for input data first

        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 256, pool=True)
        self.conv5 = conv_block(256, 256, pool=True)

        self.linear1 = nn.Linear(256*16*16, 512)   # (256*16*16, 512)
        #self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512*16*16, 512),     # (512*16*100, 512)
        #     nn.Dropout(0.2),
        #     nn.Linear(512, num_classes),
        #     nn.BatchNorm1d(num_classes),    # BN before activation
        #     nn.Sigmoid()
        # )

    def forward(self, x):           # (batch_size, 1, 256, 256)
        x = self.conv1(x)           # (batch_size, 32, 256, 256)
        x = self.conv2(x)           # (batch_size, 64, 128, 128)
        x = self.conv3(x)           # (batch_size, 128, 64, 64)
        x = self.conv4(x)           # (batch_size, 256, 32, 32)
        x = self.conv5(x)           # (batch_size, 256, 16, 16)
        x = x.view(x.size(0), -1)   # (batch_size, 256*16*16)   cannot use flatten here which will be (batch_size*256*16*16)
        x = F.relu(self.linear1(x), inplace=True)
        out = F.sigmoid(self.bn2(self.linear2(x)))
        #out = self.classifier(x)    # (512 * 16 * 100) -> (512) -> (4)
        return out.float()          # sigmoid [0 1]: return a probabilty of each class

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
