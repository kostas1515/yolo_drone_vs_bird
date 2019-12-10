from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset
    
    
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    
    prediction[:,:,:4] *= stride
    
    return prediction



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou



def get_abs_coord(box):
    
    x1 = (box[:,0] - box[:,2]/2) - 1 
    y1 = (box[:,1] - box[:,3]/2) - 1 
    x2 = (box[:,0] + box[:,2]/2) - 1 
    y2 = (box[:,1] + box[:,3]/2) - 1

    return torch.stack((x1, y1, x2, y2)).T
    

    

def get_mask(target,grid_cell,img_size,anchors):
    #lets say grid_cell is 13x13 and im_size is 416
    grid_size=img_size/grid_cell
    mask=[]
    flag_horizontal=False
    horizontal_counter=grid_size
    vertical_counter=grid_size
    lock=True


    for i in range(grid_cell):
        if(target[1]<=horizontal_counter):
            flag_horizontal=True
        horizontal_counter=grid_size+horizontal_counter
        vertical_counter=grid_size
        for j in range(grid_cell):
            if (target[0]<=vertical_counter)&flag_horizontal&lock:
                for a in range(anchors):
                    mask.append(True)
                lock=False
            else:
                for a in range(anchors):
                    mask.append(False)
            vertical_counter=grid_size+vertical_counter
    return mask


def yolo_loss(output,target):
    '''
    the targets correspon to single image,
    multiple targets can appear in the same image
    target has the size [objects,(tx,ty,tw.th,Confidence=1,class_i)]
    output should have the size [bboxes,(tx,ty,tw.th,Confidence,class_i)]
    '''

    #box size has to be torch.Size([1, grid*grid*anchors, 85])
    anchors=3#remove that make it generic
    box0=output[:,:,:].squeeze(-3)# this removes the first dimension, maybe will have to change

    xy_loss=0
    wh_loss=0
    class_loss=0
    confidence_loss=0
    total_loss=0

    #target must have size ---> torch.Size([1, obj, 85])
    #obj #torch.Size([85])
    for obj in target[0]:
        target_index=obj[0:2]# use target index to create a mask
        
        mask=get_mask(target_index,13,416,3) #these are the dimensions of grid and image


        box1=box0[mask,:]

        #box2 contains absolute coordinates
        absolute_box=get_abs_coord(box1[:,0:4])

        target_box=torch.stack([obj[0:4] for a in range(anchors)]) #range anchors!!!!
        target_box=target_box.type(torch.float)

        target_box=get_abs_coord(target_box)

        iou=bbox_iou(target_box,absolute_box)
        iou_mask=iou.max() == iou
        box1=box1[iou_mask,:]
        iou_value=iou.max()

        if (iou_value==0): #iou is 0 so bbox will be [3,6] and we only want 1 bbox
            box1=box1[0]
        else:
            box1=box1.squeeze(-2) #torch.Size([85])

        try:
            wh_loss=wh_loss+(obj[2]**(1/2)-box1[2]**(1/2))**2 + (obj[3]**(1/2)-box1[3]**(1/2))**2
        except IndexError:
            print(obj)
            print(box0)


            
        xy_loss=xy_loss+(obj[0]-box1[0])**2 + (obj[1]-box1[1])**2

        # wh_loss=wh_loss+(obj[2]**(1/2)-box1[2]**(1/2))**2 + (obj[3]**(1/2)-box1[3]**(1/2))**2

        class_loss=class_loss+((obj[5:]-box1[5:])**2).sum()

        confidence_loss =confidence_loss + (1-box1[4])**2

    total_loss=xy_loss+wh_loss+class_loss+confidence_loss

    return total_loss































    