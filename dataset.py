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
import helper as helper
import pandas as pd

class DroneDatasetCSV(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,drone_size=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        if drone_size=='large':
            new=self.dataset.apply(lambda x: helper.create_test(x = x['x'], y = x['y'], w = x['width'], h = x['height'],upper=1E10,lower=1000),axis=1)
        elif drone_size=='medium':
            new=self.dataset.apply(lambda x: helper.create_test(x = x['x'], y = x['y'], w = x['width'], h = x['height'],upper=1000,lower=100),axis=1)
        elif drone_size=='small':
            new=self.dataset.apply(lambda x: helper.create_test(x = x['x'], y = x['y'], w = x['width'], h = x['height'],upper=100,lower=0),axis=1)
        elif drone_size=='large+medium':
             new=self.dataset.apply(lambda x: helper.create_test(x = x['x'], y = x['y'], w = x['width'], h = x['height'],upper=1E10,lower=100),axis=1)
                
        if drone_size!='all':
            self.dataset['x'],self.dataset['y'],self.dataset['width'],self.dataset['height']=new[0],new[1],new[2],new[3]
            nan_value = float("NaN")
            self.dataset.replace("", nan_value, inplace=True)
            self.dataset.dropna(inplace=True)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataset.iloc[idx, 1]+'_img'+self.dataset.iloc[idx, 3].split(':')[0]+'.jpg')

        image = io.imread(img_name)
        img_width,img_height= image.shape[1],image.shape[0]
        bbox_coord = list(map(lambda x: x.split(';'),self.dataset.iloc[idx, 4:]))
        bbox_coord = np.array(bbox_coord)
        bbox_coord = bbox_coord.astype('float32').T
        bbox_coord[:,2]=(bbox_coord[:,0]+bbox_coord[:,2])/img_width
        bbox_coord[:,3]=(bbox_coord[:,3]+bbox_coord[:,1])/img_height
        bbox_coord[:,0]=(bbox_coord[:,0])/img_width
        bbox_coord[:,1]=(bbox_coord[:,1])/img_height
        
        sample = {'image': image, 'bbox_coord': bbox_coord}
        
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
        img_ = Variable(img_,requires_grad=False)# Convert to Variable
        return {'image': img_,
                'bbox_coord': torch.from_numpy(bbox_coord*self.scale)}

    
