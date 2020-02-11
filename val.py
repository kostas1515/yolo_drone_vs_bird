from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit



df = pd.read_csv('../test_annotations.csv')

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
    model.eval()
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
    




df['pred_xmin']=0.0
df['pred_ymin']=0.0
df['pred_xmax']=0.0
df['pred_ymax']=0.0
df['iou']=0.0
epochs=1
true_pos=0
counter=0
for e in range(epochs):
    for index, row in df.iterrows():
        torch.cuda.empty_cache()
        #imgpath='../images/images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        imgpath='../test_images/'+row['filename']+'_'+str(row['obj_id'])+'_img'+row['framespan'].split(':')[0]+'.jpg'
        try:
            inp = get_test_input(imgpath)
        except:
            print(imgpath)
        raw_pred = model(inp, torch.cuda.is_available())
        target=torch.tensor([[[row['x'],row['y'],row['width'],row['height'],1,1]]], dtype=torch.float)     
        target=util.xyxy_to_xywh(target)
        target=util.get_abs_coord(target)

        raw_pred=raw_pred.to(device='cuda')


        true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
        pred_mask=true_pred[0,:,4].max() == true_pred[0,:,4]
        pred_final=true_pred[:,pred_mask]

        pred_final=util.get_abs_coord(pred_final)
        pred_final[:,:,0]=pred_final[:,:,0]*1920/544
        pred_final[:,:,1]=pred_final[:,:,1]*1080/544
        pred_final[:,:,2]=pred_final[:,:,2]*1920/544
        pred_final[:,:,3]=pred_final[:,:,3]*1080/544
        
        df.pred_xmin[counter]=round(pred_final[:,:,0].item())
        df.pred_ymin[counter]=round(pred_final[:,:,1].item())
        df.pred_xmax[counter]=round(pred_final[:,:,2].item())
        df.pred_ymax[counter]=round(pred_final[:,:,3].item())
        
        iou=util.bbox_iou(target,pred_final)  
        if(iou>0.5):
            true_pos=true_pos+1
        df.iou[counter]=iou.item()
        counter=counter+1
        
print(counter)
print(true_pos/counter)

df.to_csv('test+pred_annotations.csv')
