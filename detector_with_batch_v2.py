from darknet import *
import darknet as dn
from dataset import *
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit
import torch.autograd
import torchvision.ops.boxes as nms_box


net = Darknet("../cfg/yolov3.cfg")
inp_dim=net.inp_dim
pw_ph=net.pw_ph.to(device='cuda')
cx_cy=net.cx_cy.to(device='cuda')
stride=net.stride.to(device='cuda')


'''
when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
if you start fresh then first model.load_weights and then make it parallel
'''
try:
    PATH = './test.pth'
    weights = torch.load(PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we https://pytorch.org/docs/stable/data.html#torch.utils.data.Datasetare on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(net)
        model.to(device)
        model.load_state_dict(weights)
    else:
        model=net
        model.to(device)
        model.load_state_dict(weights)
        
except FileNotFoundError: 
    net.load_weights("../yolov3.weights")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(net)
        model.to(device)
    else:
        model=net

    
drone_size='all'
print('training for '+ drone_size+'\n')
transformed_dataset=DroneDatasetCSV(csv_file='../annotations.csv',
                                           root_dir='../images/images/',
                                           drone_size=drone_size,
                                           transform=transforms.Compose([
                                               ResizeToTensor(inp_dim)
                                           ]))


dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=20

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.005)
epochs=150
total_loss=0
write=0
misses=0
break_flag=0
avg_iou=0
for e in range(epochs):
    prg_counter=0
    train_counter=0
    total_loss=0
    avg_iou=0
    avg_infs=0
    print("\n epoch "+str(e))
    misses=0
    for i_batch, sample_batched in enumerate(dataloader):
        write=0
        optimizer.zero_grad()
        raw_pred = model(sample_batched['image'], torch.cuda.is_available())
        
        target=sample_batched['bbox_coord'].unsqueeze(-3)
        target=target.to(device='cuda')
        for b in range(sample_batched['image'].shape[0]):
            if (write==0):
                anchors=pw_ph
                offset=cx_cy
                strd=stride
                write=1
            else:
                anchors=torch.cat((anchors,pw_ph),0).to(device='cuda')
                offset=torch.cat((offset,cx_cy),0).to(device='cuda')
                strd=torch.cat((strd,stride),0).to(device='cuda')
                
        true_pred=util.transform(raw_pred.clone(),anchors,offset,strd)
        iou_mask,noobj_mask=util.get_responsible_masks(true_pred,target,offset,stride)
        
        iou=torch.diag(util.bbox_iou(util.get_abs_coord(true_pred[iou_mask.T,:].unsqueeze(-3)),target)).mean().item()
        
        
        noobj_box=raw_pred[:,:,4:5].clone()
        conf=noobj_box[iou_mask.T,:].mean().item()
        
        noobj_box=noobj_box[noobj_mask.T,:]
        no_obj_conf=noobj_box.mean().item()
        
        raw_pred=raw_pred[iou_mask.T,:]
        anchors=anchors[iou_mask.T,:]
        offset=offset[iou_mask.T,:]
        strd=strd[iou_mask.T,:]
        
        if(strd.shape[0]==sample_batched['image'].shape[0]):#this means that iou_mask failed and was all true, because max of zeros is true for all lenght of mask strd
            target=util.xyxy_to_xywh(target)
            target=target.squeeze(1)
            target=util.transform_groundtruth(target,anchors,offset,strd)
            loss=util.yolo_loss(raw_pred,target,noobj_box,batch_size)
            loss.backward()
            optimizer.step()
            total_loss=total_loss+loss.item()
            avg_iou=avg_iou+iou
            sys.stdout.write('\r Progress is ' +str(prg_counter/dataset_len*100*batch_size)+'%' ' loss is: '+ str(loss.item()))
            sys.stdout.write(' Iou is ' +str(iou)+' conf is '+str(conf)+ ' no_obj conf is '+str(no_obj_conf))
            sys.stdout.flush()
            del loss, raw_pred, target, true_pred, sample_batched['image'],iou,noobj_box,conf
            torch.cuda.empty_cache()
            prg_counter=prg_counter+1
            train_counter=train_counter+1
        else:
            misses=misses+1
            print('missed')
            print(strd.shape[0])
            prg_counter=prg_counter+1
    torch.save(model.state_dict(), PATH)
    print('\ntotal number of misses is ' + str(misses))
    print('\n total average loss is '+str(total_loss/train_counter))
    print('\n total average iou is '+str(avg_iou/train_counter))