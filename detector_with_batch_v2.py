from darknet import *
import darknet as dn
from dataset import *
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit


net = Darknet("../cfg/yolov3.cfg")
inp_dim=net.inp_dim.to(device='cuda')
pw_ph=net.pw_ph.to(device='cuda')
cx_cy=net.cx_cy.to(device='cuda')
stride=net.stride.to(device='cuda')


'''
when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
if you start fresh then first model.load_weights and then make it parallel
'''
try:
    PATH = './local2.pth'
    weights = torch.load(PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(net)

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
    

transformed_dataset=DroneDatasetCSV(csv_file='../annotations.csv',
                                           root_dir='../images/images/',
                                           transform=transforms.Compose([
                                               ResizeToTensor(544)
                                           ]))



batch_size=16

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs=20
total_loss=0
write=0
for e in range(epochs):
    prg_counter=0
    total_loss=0
    print("\n epoch "+str(e))
    for i_batch, sample_batched in enumerate(dataloader):
        write=0
        raw_pred = model(sample_batched['image'], torch.cuda.is_available())
        target=util.xyxy_to_xywh(sample_batched['bbox_coord'].unsqueeze(-3))

        target=target.to(device='cuda')
        raw_pred=raw_pred.to(device='cuda')
        
        for b in range(batch_size):
            if (write==0):
                anchors=pw_ph
                offset=cx_cy
                strd=stride
                write=1
            else:
                anchors=torch.cat((anchors,pw_ph),0)
                offset=torch.cat((offset,cx_cy),0)
                strd=torch.cat((strd,stride),0)
        
        true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
        iou_mask,noobj_mask=util.get_responsible_masks(true_pred,target)
            

        noobj_box=raw_pred[:,:,4:5].clone()
        noobj_box=noobj_box[noobj_mask.T,:]
        raw_pred=raw_pred[iou_mask.T,:]
        anchors=anchors[iou_mask.T,:]
        offset=offset[iou_mask.T,:]
        strd=strd[iou_mask.T,:]
        
        if(strd.shape[0]==batch_size):#this means that iou_mask failed and was all true, because max of zeros is true for all lenght of mask strd
            target=target.squeeze(-2)
            target=target/strd
            target=util.transform_groundtruth(target,anchors,offset)

            loss=util.yolo_loss(raw_pred,target,noobj_box)

            loss.backward()
            optimizer.step()
            total_loss=total_loss+loss.item()
            sys.stdout.write('\r Progress is ' +str(prg_counter/9570*100*batch_size)+'%' ' loss is: '+ str(loss.item()))
            sys.stdout.flush()
            del loss, raw_pred
            torch.cuda.empty_cache()
            prg_counter=prg_counter+1
        else:
            print('missed')
            print(strd.shape[0])
            print(targets.shape)
            print(targets)
            print(imgpath)
            prg_counter=prg_counter+1
                
    torch.save(model.state_dict(), PATH)
    print('\n total average loss is '+str(total_loss/9570*64))