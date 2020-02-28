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
    PATH = './l+m2_normal.pth'
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

    
drone_size='large+medium'
print('training for '+ drone_size+'\n')
transformed_dataset=DroneDatasetCSV(csv_file='../annotations.csv',
                                           root_dir='../images/images/',
                                           drone_size=drone_size,
                                           transform=transforms.Compose([
                                               ResizeToTensor(inp_dim)
                                           ]))


dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=8

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


optimizer = optim.Adam(model.parameters(), lr=0.000001)
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
        sample_batched['image'],sample_batched['bbox_coord']=sample_batched['image'].to(device='cuda'),sample_batched['bbox_coord'].to(device='cuda')
        raw_pred = model(sample_batched['image'], torch.cuda.is_available())
        
        target=util.xyxy_to_xywh(sample_batched['bbox_coord'].unsqueeze(-3))
        
        
        target=target.to(device='cuda')
        raw_pred=raw_pred.to(device='cuda')
        
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
        
        true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride,inp_dim)
        iou_mask,noobj_mask=util.get_responsible_masks(true_pred,target)
        
        abs_pred=util.get_abs_coord(true_pred)
        abs_true=util.get_abs_coord(target)
        iou=util.bbox_iou(abs_pred,abs_true,True)
        
        iou_mask=iou.max(dim=0)[0] == iou
        infs=iou.ne(iou).sum().item()
        iou[iou.ne(iou)] = 0 #removes nan values
        

        noobj_box=raw_pred[:,:,4:5].clone()
        noobj_box=noobj_box[noobj_mask.T,:]
        raw_pred=raw_pred[iou_mask.T,:]
        anchors=anchors[iou_mask.T,:]
        offset=offset[iou_mask.T,:]
        strd=strd[iou_mask.T,:]
        
        if(strd.shape[0]==sample_batched['image'].shape[0]):#this means that iou_mask failed and was all true, because max of zeros is true for all lenght of mask strd
            target=target.squeeze(-2)

            target=target*inp_dim/strd
            target=util.transform_groundtruth(target,anchors,offset)
            loss=util.yolo_loss(raw_pred,target,noobj_box,batch_size)
            loss.backward()
            optimizer.step()
            total_loss=total_loss+loss.item()
            sys.stdout.write('\r Progress is ' +str(prg_counter/dataset_len*100*batch_size)+'%' ' loss is: '+ str(loss.item()))
            sys.stdout.write(' IoU is: ' + str(iou.max(dim=0)[0].mean().item())+' infs are: '+str(infs))
            sys.stdout.flush()
            del loss, raw_pred, target, true_pred, sample_batched['image']
            avg_iou=avg_iou+iou.max(dim=0)[0].mean().data#use .data in =+ statements 
            avg_infs=avg_infs+infs
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
    print('\n total average infs is '+str(avg_infs/train_counter))