from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit



df = pd.read_csv('../annotations.csv')

net = Darknet("../cfg/yolov3.cfg")
inp_dim=net.inp_dim
pw_ph=net.pw_ph
cx_cy=net.cx_cy
stride=net.stride


'''
when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
if you start fresh then first model.load_weights and then make it parallel
'''
try:
    PATH = './darknet.pth'
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
    




optimizer = optim.Adam(model.parameters(), lr=0.00001)


epochs=20


lock=0
total_loss=0
batch_counter=0
batch_loss=0
for e in range(epochs):
    prg_counter=0
    total_loss=0
    print("\n epoch "+str(e))
    for index, row in df.iterrows():
        optimizer.zero_grad()
        imgpath='../images/images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        inp = get_test_input(imgpath)
        
        target=torch.tensor([[[row['x']/(1920),row['y']/(1080),row['width']/(1920),row['height']/(1080),1,1]]])
        target= (target.squeeze(-3))     
        target=util.xyxy_to_xywh(target) # target in normal scale
        
        
        raw_pred = model(inp, torch.cuda.is_available())

        target=target.to(device='cuda')
        raw_pred=raw_pred.to(device='cuda')
        

        true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
        true_pred=true_pred.squeeze(-3)
        iou_mask,noobj_mask=util.get_responsible_masks(true_pred,target*inp_dim)
        
        raw_pred=raw_pred.squeeze(-3)
        
        anchors=pw_ph.clone().squeeze(-3)
        offset=cx_cy.clone().squeeze(-3)
        strd=stride.clone().squeeze(-3)
        
        noobj_box=raw_pred[:,4].clone()
        noobj_box=noobj_box[noobj_mask]
        
        raw_pred=raw_pred[iou_mask,:]
        anchors=anchors[iou_mask,:]
        offset=offset[iou_mask,:]
        strd=strd[iou_mask,:]
        
        if(strd.shape[0]==1):
            target[:,0:4]=target[:,0:4]*(inp_dim/strd)
            target=util.transform_groundtruth(target,anchors,offset)
            if(batch_counter<63):
                loss=util.yolo_loss(raw_pred,target,noobj_box)
                batch_loss=batch_loss+loss.item()
                batch_counter=batch_counter+1

            else:
                batch_counter=batch_counter+1
                loss=util.yolo_loss(raw_pred,target,noobj_box)
                batch_loss=(batch_loss+loss)/batch_counter
                print(batch_counter)
                batch_counter=0
                batch_loss.backward()
                optimizer.step()
                total_loss=total_loss+batch_loss.item()
                sys.stdout.write('\r Progress is ' +str(prg_counter/9570*100)+'%' ' loss is: '+ str(batch_loss.item()))
                batch_loss=0
                sys.stdout.flush()
            prg_counter=prg_counter+1
        else:
            print('missed')
            prg_counter=prg_counter+1
    torch.save(model.state_dict(), PATH)
    print('\n total average loss is '+str(total_loss/9570*64))