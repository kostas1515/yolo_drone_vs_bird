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
    




optimizer = optim.Adam(model.parameters(), lr=0.00001)


epochs=40


lock=0
total_loss=0
batch_counter=0
batch_loss=0


anchors=torch.empty(pw_ph.shape).cuda()
offset=torch.empty(cx_cy.shape).cuda()
strd=torch.empty(stride.shape).cuda()
inputs=torch.empty(3,544,544)
targets=torch.empty(6)
write=0
batch_size=8
batch_counter=0
for e in range(epochs):
    prg_counter=0
    total_loss=0
    print("\n epoch "+str(e))
    for index, row in df.iterrows():
        batch_counter=batch_counter+1
        optimizer.zero_grad()
        imgpath='../images/images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        inp = get_test_input(imgpath)
        
        target=torch.tensor([[[row['x']/(1920),row['y']/(1080),row['width']/(1920),row['height']/(1080),1,1]]])
        
        
        if (write==0):
            targets=target
            inputs=inp
            anchors=pw_ph
            offset=cx_cy
            strd=stride
            write=1
        else:
            targets=torch.cat((targets,target),1)
            inputs=torch.cat((inputs,inp),0)
            anchors=torch.cat((anchors,pw_ph),0)
            offset=torch.cat((offset,cx_cy),0)
            strd=torch.cat((strd,stride),0)
        
        if(batch_counter==batch_size):
            batch_counter=0
            write=0
            targets=util.xyxy_to_xywh(targets) # target in normal scale
            raw_pred = model(inputs, torch.cuda.is_available())

            targets=targets.to(device='cuda')
            raw_pred=raw_pred.to(device='cuda')
        

            true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
            iou_mask,noobj_mask=util.get_responsible_masks(true_pred,target*inp_dim)
            

            noobj_box=raw_pred[:,:,4:5].clone()
            noobj_box=noobj_box[noobj_mask.T,:]
            raw_pred=raw_pred[iou_mask.T,:]
            anchors=anchors[iou_mask.T,:]
            offset=offset[iou_mask.T,:]
            strd=strd[iou_mask.T,:]
        

        
            if(strd.shape[0]==batch_size):#this means that iou_mask failed and was all true, because max of zeros is true for all lenght of mask strd
                targets=targets.squeeze(-2)
                targets=targets*(inp_dim/strd)
                targets=util.transform_groundtruth(targets,anchors,offset)

                loss=util.yolo_loss(raw_pred,targets,noobj_box,batch_size)
                
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