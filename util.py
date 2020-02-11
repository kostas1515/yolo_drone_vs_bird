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
    strd=torch.ones(1,anchors.shape[1],1)*stride
    
    return prediction,anchors,x_y_offset,strd


def transform(prediction,anchors,x_y_offset,stride,CUDA = True):
    '''
    This function takes the raw predicted output from yolo last layer in the correct
    '[batch_size,3*grid*grid,4+1+class_num] * grid_scale' size and transforms it into the real world coordinates
    Inputs: raw prediction, xy_offset, anchors, stride
    Output: real world prediction
    '''
    #Sigmoid the  centre_X, centre_Y.
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    
    #Add the center offsets
    prediction[:,:,:2] += x_y_offset
    
    #log space transform height and the width
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,:4] *= stride
    
    return prediction

def predict(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    this function reorders 4 coordinates tx,ty,tw,th as well as confidence and class probabilities
    then it sigmoids the confidence and the class probabilites
    Inputs: raw predictions from yolo last layer
    Outputs: pred: raw coordinate prediction, sigmoided confidence and class probabilities
    size of pred= [batch_size,3*grid*grid,4+1+class_num] in 3 different scales: grid, 2*gird,4*grid concatenated
    it also return stride, anchors and xy_offset in the same format to use later to transform raw output
    in the real world coordinates
    '''
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    
    #Sigmoid object confidencce
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    
    
    return prediction



def get_utillities(stride,inp_dim, anchors, num_classes):
    '''
    this function reorders 4 coordinates tx,ty,tw,th as well as confidence and class probabilities
    then it sigmoids the confidence and the class probabilites
    Inputs: raw predictions from yolo last layer
    Outputs: pred: raw coordinate prediction, sigmoided confidence and class probabilities
    size of pred= [batch_size,3*grid*grid,4+1+class_num] in 3 different scales: grid, 2*gird,4*grid concatenated
    it also return stride, anchors and xy_offset in the same format to use later to transform raw output
    in the real world coordinates
    '''
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)


    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    
    strd=torch.ones(1,anchors.shape[1],1)*stride
    
    return anchors,x_y_offset,strd
    



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,:,0], box1[:,:,1], box1[:,:,2], box1[:,:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,:,0], box2[:,:,1], box2[:,:,2], box2[:,:,3]
    
    #get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 ,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 , torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 ,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 , torch.zeros(inter_rect_x2.shape))
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 )*(b1_y2 - b1_y1 )
    b2_area = (b2_x2 - b2_x1 )*(b2_y2 - b2_y1 )
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def get_utils(stride,inp_dim, anchors, num_classes, CUDA = True):
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    
    strd=torch.ones(1,anchors.shape[1],1)*stride
    
    return anchors,x_y_offset,strd
    

# def get_abs_target_coord(box):
#     #the target coord are measured from top-left
#     x1 = box[:,0]
#     y1 = box[:,1]
#     x2 = box[:,0] + box[:,2]
#     y2 = box[:,1] + box[:,3]

#     return torch.stack((x1, y1, x2, y2)).T

def get_abs_coord(box):
    # yolo predicts center coordinates
    box=box.cuda()
    x1 = (box[:,:,0] - box[:,:,2]/2) 
    y1 = (box[:,:,1] - box[:,:,3]/2) 
    x2 = (box[:,:,0] + box[:,:,2]/2) 
    y2 = (box[:,:,1] + box[:,:,3]/2) 
    
    return torch.stack((x1, y1, x2, y2)).T

def xyxy_to_xywh(box):
    box=box.cuda()
    xc = (box[:,:,2]/2+ box[:,:,0])
    yc = (box[:,:,3]/2+ box[:,:,1])
    
    w = (box[:,:,2])
    h = (box[:,:,3])
    
    return torch.stack((xc, yc, w, h)).T


def get_responsible_masks(transformed_output,target):
    '''
    this function takes the transformed_output and
    the target box in respect to the resized image size
    and returns a mask which can be applied to select the 
    best raw input,anchors and cx_cy_offset
    and the noobj_mask for the negatives
    '''
    abs_pred_coord=get_abs_coord(transformed_output)
    abs_target_coord=get_abs_coord(target)
    iou=bbox_iou(abs_pred_coord,abs_target_coord)
    iou_mask=iou.max(dim=0)[0] == iou
    
    ignore_mask=0.5>iou
    inverted_mask=iou.max(dim=0)[0] != iou
    noobj_mask=ignore_mask & inverted_mask
    
    return iou_mask,noobj_mask

    
def transform_groundtruth(target,anchors,cx_cy):
    '''
    this function takes the target real coordinates and transfroms them into grid cell coordinates
    returns the groundtruth to use for optimisation step
    '''
    target[:,0:2]=target[:,0:2]-cx_cy
    target[:,2:4]=torch.log(target[:,2:4]/anchors)
    
    return target

def yolo_loss(output,obj,noobj_box):
    '''
    the targets correspon to single image,
    multiple targets can appear in the same image
    target has the size [objects,(tx,ty,tw.th,Confidence=1,class_i)]
    output should have the size [bboxes,(tx,ty,tw.th,Confidence,class_i)]
    inp_dim is the widht and height of the image specified in yolov3.cfg
    '''

    #box size has to be torch.Size([1, grid*grid*anchors, 6])
#     box0=output[:,:,:].squeeze(-3)# this removes the first dimension, maybe will have to change
    
    #box0[box0.ne(box0)] = 0 # this substitute all nan with 0
    xy_loss=0
    wh_loss=0
    class_loss=0
    confidence_loss=0
    total_loss=0
    no_obj_conf_loss=0
    no_obj_counter=0
    #target must have size ---> torch.Size([1, obj, ])
    #obj #torch.Size([85]):
        #abs_target contains xmin ymin xmax ymax coord     
        #abs_pred_box contains xmin ymin xmax ymax coord        

    wh_loss=wh_loss+(obj[:,2]-output[:,2])**2 + (obj[:,3]-output[:,3])**2
                   
    xy_loss=xy_loss+(obj[:,0]-output[:,0])**2 + (obj[:,1]-output[:,1])**2

    class_loss=class_loss+(1-output[:,5])
        
        #the confidense penalty could be either 1 or the actual IoU
    confidence_loss =confidence_loss + (1-output[:,4])**2 

    no_obj_conf_loss =no_obj_conf_loss + (0-noobj_box[:,0])**2
        
    total_loss=5*xy_loss.mean()+5*wh_loss.mean()+class_loss.mean()+confidence_loss.sum()+0.5*no_obj_conf_loss.sum()
    
    return total_loss































    