import os
import cv2
import numpy as np
import pandas as pd

# Dataset, dataloader
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.transforms import functional as TF

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
    def rle2mask(self, rle, imgshape = (256,1600)):
        width = imgshape[0]
        height= imgshape[1]
        
        mask= np.zeros( width*height ).astype(np.uint8)
        
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]
            
        return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )

    def build_mask(self, rles, input_shape = (256,1600)):
        depth = len(rles)
        height, width = input_shape
        masks = np.zeros((height, width, depth))
        
        for i, rle in enumerate(rles):
            if type(rle) is str:
                masks[:, :, i] = self.rle2mask(rle, (height, width))
        
        return masks

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
            
        traincsv_path = os.path.join(root_dir, 'data', 'train_rle_pivot.csv')
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

    def transform(self, image, mask):
        # To tensor (image, mask)
        image = TF.to_tensor(image) #Before tensor: (256, 1600, 3)
        mask = TF.to_tensor(mask)   #After tensor: (3, 256, 1600)
        
        # # Resize (image, mask)
        # resize = transforms.Resize(size=(int(image.shape[0]), int(image.shape[0])))
        # image = resize(image)
        # mask = resize(mask)

        # Grayscale (image)
        image = TF.rgb_to_grayscale(image, num_output_channels=1)   #After grayscale: (1, 256, 1600)
        
        # RandomRotation (image, mask)
        angle = transforms.RandomRotation(degrees=1).get_params(degrees=[-1,1])
        image_rotated = TF.rotate(image, angle)
        mask_rotated = TF.rotate(mask, angle)
        
        return image_rotated, image, mask_rotated
    
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
        # Label
        label = self.df.iloc[idx, 1:].notnull().values.astype(float)
        # mask
        rles = self.df.iloc[idx, 1:].values
        mask = self.build_mask(rles)

        image_rotated, image, mask_rotated = self.transform(image, mask)
        
        if self.train:
            sample = {
                'image': image_rotated,           # (1, 256, 1600), do not squeeze here as conv layer require 3D input
                'label': torch.tensor(label),
                'mask': mask_rotated                    # (4, 256, 1600)
            }       
        else:
            sample = {
                'ImageId': self.df.loc[idx, 'ImageId'],
                'image': image
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

    dl = ImageDataLoader(validation_split=0.2, batch_size=8, shuffle=True)
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
    for step, data in enumerate(train_loader):
        images = Variable(data['image'])
        labels = Variable(data['label'])
        masks = Variable(data['mask'])

        image = images[0].detach().numpy()
        label = labels[0].detach().numpy()
        mask = masks[0].detach().numpy()   

        plt.figure(figsize=(12,12))
        plt.subplot(511)
        plt.axis('off')
        plt.title(f'image: {image.shape}, label: {label}')
        plt.imshow(image.squeeze()) # Always use .detach() instead of .data which will be expired
        for i in range(mask.shape[0]):
            plt.subplot(512+i)
            plt.title(f'mask{i+1}: {mask.shape}')
            plt.imshow(mask[i].squeeze())
            plt.axis('off')
        plt.show()
        break
    
    # test_loader
    for step, data in enumerate(test_loader):
        imgids = data['ImageId']
        images = Variable(data['image'])

        imgid = imgids[0]
        image = images[0].detach().numpy()

        plt.figure(figsize=(12,4))
        plt.axis('off')
        plt.title(f'{imgid}: {image.shape}')
        plt.imshow(image.squeeze()) # Always use .detach() instead of .data which will be expired
        plt.show()
        break