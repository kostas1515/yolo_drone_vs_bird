from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
   
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
    
    prediction[:,:,:2] = prediction[:,:,:2]*(stride)
    #log space transform height and the width
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]).clamp(max=1E4)*anchors*stride
    
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
    
    
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    
    strd=torch.ones(1,anchors.shape[1],1)*stride
    
    return anchors,x_y_offset,strd
    



def bbox_iou(box1, box2,CUDA=True):
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
    if CUDA:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 ,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 , torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 ,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 , torch.zeros(inter_rect_x2.shape))
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 )*(b1_y2 - b1_y1 )
    b2_area = (b2_x2 - b2_x1 )*(b2_y2 - b2_y1 )
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

    
def get_abs_coord(box):
    # yolo predicts center coordinates
    if torch.cuda.is_available():
        box=box.cuda()
    x1 = (box[:,:,0] - box[:,:,2]/2) 
    y1 = (box[:,:,1] - box[:,:,3]/2) 
    x2 = (box[:,:,0] + box[:,:,2]/2) 
    y2 = (box[:,:,1] + box[:,:,3]/2) 
    
    return torch.stack((x1, y1, x2, y2)).T

def xyxy_to_xywh(box):
    if torch.cuda.is_available():
        box=box.cuda()
    xc = (box[:,:,2]- box[:,:,0])/2 +box[:,:,0]
    yc = (box[:,:,3]- box[:,:,1])/2 +box[:,:,1]
    
    w = (box[:,:,2]- box[:,:,0])
    h = (box[:,:,3]- box[:,:,1])
    
    return torch.stack((xc, yc, w, h)).T

def same_picture_mask(responsible_mask,mask):
    '''
    mask is a list containing the number of objects per image
    '''
    k=0
    for i,count in enumerate(mask):
        same_image_mask=False
        for obj in range(count):
            same_image_mask=same_image_mask+responsible_mask[:,k+obj]
        for obj in range(count):
            responsible_mask[:,k+obj]=same_image_mask
        k=k+count
    return responsible_mask


def get_responsible_masks(transformed_output,targets,offset,strd,mask):
    '''
    this function takes the transformed_output and
    the target box in respect to the resized image size
    and returns a mask which can be applied to select the 
    best raw input,anchors and cx_cy_offset
    and the noobj_mask for the negatives
    targets is a list
    '''
    #first compute the centered target coords
    centered_target=xyxy_to_xywh(targets)[:,:,0:2]
    #then devide by stride to get the relative grid size coordinates, floor the result to get the corresponding cell
    centered_target=torch.floor(centered_target/strd)
    
    #create a mask to find where the gt falls into which gridcell in the grid coordinate system
    fall_into_mask=centered_target==offset
    fall_into_mask=fall_into_mask[:,:,0]&fall_into_mask[:,:,1]
#     fall_into_mask= ~fall_into_mask
    #create a copy of the transformed output
    best_bboxes=transformed_output.clone()
    #apply reverse mask to copy in order to zero all other bbox locations
    best_bboxes[~fall_into_mask]=0   
    #transform the copy to xmin,xmax,ymin,ymax
    best_responsible_coord=get_abs_coord(best_bboxes)
    #calculate best iou and mask
    responsible_iou=bbox_iou(best_responsible_coord,targets,True)

    responsible_iou[responsible_iou.ne(responsible_iou)] = 0
    responsible_mask=responsible_iou.max(dim=0)[0] == responsible_iou
    abs_coord=get_abs_coord(transformed_output)
    iou=bbox_iou(abs_coord,targets,True)
    iou[iou.ne(iou)] = 0
    ignore_mask=0.5>iou
    inverted_mask=iou.max(dim=0)[0] != iou
    noobj_mask=ignore_mask & inverted_mask & ~same_picture_mask(responsible_mask.clone(),mask)
    
    return responsible_mask,noobj_mask

    
def transform_groundtruth(target,anchors,cx_cy,strd):
    '''
    this function takes the target real coordinates and transfroms them into grid cell coordinates
    returns the groundtruth to use for optimisation step
    consider using sigmoid to prediction, insted of inversing groundtruth
    '''
    target=target/strd
    target[:,0:2]=target[:,0:2]-cx_cy
    target[:,0:2]=torch.log(target[:,0:2]/(1-target[:,0:2])).clamp(min=-10, max=10)
    target[:,2:4]=torch.log(target[:,2:4]/anchors)
    
    return target

def yolo_loss(output,obj,noobj_box,batch_size):
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
    gamma=1
    alpha=0.95

    wh_loss=wh_loss+(obj[:,2]-output[:,2])**2 + (obj[:,3]-output[:,3])**2
                   
    xy_loss=xy_loss+(obj[:,0]-output[:,0])**2 + (obj[:,1]-output[:,1])**2

    #class_loss=class_loss+(1-output[:,5])
        
        #the confidense penalty could be either 1 or the actual IoU
    confidence_loss =confidence_loss -alpha*((1-output[:,4])**gamma)*torch.log(output[:,4])

    no_obj_conf_loss =no_obj_conf_loss -(1-alpha)*(noobj_box[:,0]**gamma)*torch.log(1-noobj_box[:,0])
    
#     confidence_loss =confidence_loss +(1-output[:,4])**2

#     no_obj_conf_loss =no_obj_conf_loss +(noobj_box[:,0])**2
        
    total_loss=5*xy_loss.mean()+5*wh_loss.mean()+confidence_loss.sum()+no_obj_conf_loss.sum()/batch_size
    
    return total_loss































    