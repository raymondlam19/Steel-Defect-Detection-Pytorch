import os
import cv2
import numpy as np
import pandas as pd

# Dataset, dataloader
import torch
import torch.utils.data as Data
from torchvision import transforms

try:
    from utils import ROOT_DIR
except:
    print('testing: data_loaders.py')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ImageDataset(Data.Dataset):
    def __init__(self, train=True):
        """
        Initialize a dataset as a list of IDs corresponding to each item of data set
        Args:
            img_dir (string): Directory with all the images.
            csv_file (string): filename of the csv file with annotations.
            train_test (string): Input 'train'/'test' to indicate the img folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        try:
            root_dir = ROOT_DIR
        except:
            root_dir = os.path.join(os.path.dirname(__file__), '..')
            
        traincsv_path = os.path.join(root_dir, 'data', 'train_pivot.csv')
        testcsv_path = os.path.join(root_dir, 'data', 'testset.csv')
        trainimg_dir = os.path.join(root_dir, 'data', 'train_images')
        testimg_dir = os.path.join(root_dir, 'data', 'test_images')
        
        self.train = train
        self.df = pd.read_csv(traincsv_path if self.train else testcsv_path)
        self.img_dir = trainimg_dir if self.train else testimg_dir

    def __len__(self):
        """
        Return the length of data set using list of IDs
        :return: number of samples in data set
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Generate one item of data set.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Image
        img_name = os.path.join(self.img_dir, self.df.loc[idx, 'ImageId'])
        image = cv2.imread(img_name)
        
        # Transform image
        transform = transforms.Compose([
            transforms.ToTensor(),                          # Auto turn to range [0 1] in .ToTensor()
            transforms.Grayscale(num_output_channels=1),    # ToTensor first as .Grayscale() require tensor input
            transforms.Resize([int(image.shape[0]),int(image.shape[0])]),   # resize make training faster
            transforms.RandomRotation(degrees=1),
            #transforms.Normalize(mean=[0.5], std=[0.5])    # Why 0.5? -> We will use BN at the beginning of the model instead
        ])
        
        # Label
        label = self.df.iloc[idx, 1:].values.astype(float)
        
        if self.train:
            sample = {
                'image': transform(image),              # (1, 256, 1600), do not squeeze here as conv layer require 3D input
                'label': torch.tensor(label).float()    # sync label to float (pred from model will be make into float as well)
            }       
        else:
            sample = {
                'ImageId': self.df.loc[idx, 'ImageId'],
                'image': transform(image)
            }
        return dotdict(sample)

class ImageDataLoader(Data.DataLoader):
    def __init__(self, validation_split, batch_size, shuffle=True):
        self.dataset = ImageDataset(train=True)
        self.train_dataset, self.val_dataset = Data.random_split(self.dataset, [int(len(self.dataset)*(1-validation_split)), len(self.dataset)-int(len(self.dataset)*(1-validation_split))])
        self.test_dataset = ImageDataset(train=False)

        self.train_loader = Data.DataLoader(
            self.train_dataset,
            batch_size = batch_size, 
            shuffle = shuffle
        )
        self.val_loader = Data.DataLoader(
            self.val_dataset,
            batch_size = batch_size, 
            shuffle = shuffle
        )
        self.test_loader = Data.DataLoader(
            self.test_dataset,
            batch_size = batch_size, 
            shuffle = False
        )
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.autograd import Variable

    dl = ImageDataLoader(validation_split=0.2, batch_size=16, shuffle=True)
    train_loader = dl.train_loader
    print(f"Train set length: {len(train_loader.dataset)}")
    print(f"Total training steps in an epoch: {len(train_loader)}\n")

    val_loader = dl.val_loader
    print(f"Val set length: {len(val_loader.dataset)}")
    print(f"Total val steps in an epoch: {len(val_loader)}\n")

    test_loader = dl.test_loader
    print(f"Test set length: {len(test_loader.dataset)}")
    print(f"Total testing steps in an epoch: {len(test_loader)}\n")

    # Show a sample image with its label (train_loader)
    # for step, data in enumerate(train_loader):
    #     b_x = Variable(data['image'])
    #     b_y = Variable(data['label'])

    #     plt.figure(figsize=(12,4))
    #     plt.axis('off')
    #     plt.imshow(b_x[0].detach().numpy().squeeze()) # Always use .detach() instead of .data which will be expired
    #     plt.show()
    #     print(b_y[0].detach().numpy())
    #     break
    
    # test_loader
    for step, data in enumerate(test_loader):
        imgids = data['ImageId']
        images = Variable(data['image'])

        plt.figure(figsize=(12,4))
        plt.axis('off')
        plt.title(imgids[0])
        plt.imshow(images[0].detach().numpy().squeeze()) # Always use .detach() instead of .data which will be expired
        plt.show()
        break


# from torchvision import datasets, transforms
# from base import BaseDataLoader


# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
