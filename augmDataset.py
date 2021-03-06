from __future__ import print_function, division
from torch.autograd import Variable
import numpy as np
import cv2
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import tarfile

class AugDatasetCSV(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file,archieve, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_csv(csv_file)
        self.dataset=self.dataset.sort_values(by=['filename'])
        self.root_dir = root_dir
        self.tar= tarfile.open(archieve, "r")
        self.tar.next()
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        tar_img=self.tar.next() #member
        
        img_name = os.path.join(self.root_dir,
                                self.dataset.iloc[idx, 5].split('.')[0]+'.jpg')
        if('/'+tar_img.name==img_name):
            
            extracted_image=self.tar.extractfile(tar_img)
            
            image = io.imread(extracted_image)
            
            img_width,img_height= image.shape[1],image.shape[0]
            bbox_coord = self.dataset.iloc[idx,1:5]
            bbox_coord = np.array([bbox_coord])
            bbox_coord = bbox_coord.astype('float32').reshape(4)
            bbox_coord[0]=bbox_coord[0]/img_width
            bbox_coord[1]=bbox_coord[1]/img_height
            bbox_coord[2]=bbox_coord[2]/img_width
            bbox_coord[3]=bbox_coord[3]/img_height
            sample = {'image': image, 'bbox_coord': bbox_coord}
        else:
            print('false')

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ResizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img, bbox_coord = sample['image'], sample['bbox_coord']

        img = cv2.resize(img, (self.scale,self.scale))          #Resize to the input dimension
        img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
        img_ = img_/255.0       #Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()     #Convert to float
        img_ = Variable(img_,requires_grad=False)                     # Convert to Variable
        return {'image': img_,
                'bbox_coord': torch.from_numpy(bbox_coord*self.scale)}
