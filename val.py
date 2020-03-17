from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit
from dataset import *
import torchvision.ops.boxes as nms_box
import helper as helper

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
    PATH = './test.pth'
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
drone_size='all'
print('testing for '+ drone_size+'\n')
transformed_dataset=DroneDatasetCSV(csv_file='../test_annotations.csv',
                                           root_dir='../test_images/',
                                           drone_size=drone_size,
                                           transform=transforms.Compose([
                                               ResizeToTensor(inp_dim)
                                           ]))


dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=1

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=False,collate_fn=helper.my_collate, num_workers=0)

true_pos=0
false_pos=0
counter=0
iou_threshold=0
confidence=0.7
recall_counter=0

for images,targets in dataloader:
    inp=images.cuda()
    raw_pred = model(inp, torch.cuda.is_available())
    targets,anchors,offset,strd,mask=helper.collapse_boxes(targets,pw_ph,cx_cy,stride)
    
    targets=targets

    raw_pred=raw_pred.to(device='cuda')
    true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
    sorted_pred=torch.sort(true_pred[0,:,4],descending=True)
    pred_mask=sorted_pred[0]>confidence
    
    indices=(sorted_pred[1][pred_mask])
    pred_final=true_pred[0,indices,:]
#     pred_mask=true_pred[0,:,4].max() == true_pred[0,:,4]
    pred_final_coord=util.get_abs_coord(pred_final.unsqueeze(-2))
    
#       df.pred_xmin[counter]=round(pred_final[:,:,0].item())
#       df.pred_ymin[counter]=round(pred_final[:,:,1].item())
#       df.pred_xmax[counter]=round(pred_final[:,:,2].item())
#       df.pred_ymax[counter]=round(pred_final[:,:,3].item())
    
    indices=nms_box.nms(pred_final_coord[0],pred_final[:,4],iou_threshold)
    pred_final_coord=pred_final_coord.to('cuda')
    targets=targets.to('cuda')
    pred_final_coord=(pred_final_coord[:,indices]).squeeze(0)
    if(len(pred_final_coord.size())!=0):
        iou=nms_box.box_iou(targets,pred_final_coord)
        true_pos=true_pos+((iou>=0.5).sum(dim=0)).sum().item()
        false_pos=false_pos+((iou<0.5).sum(dim=0)).sum().item()
    print(iou)
    counter=counter+targets.shape[0]
print('precision')
precision=true_pos/(true_pos+false_pos)
print(precision)

print('recall')
recall=true_pos/(counter)
print(recall)
f1=2*(precision*recall)/(precision+recall)
print(f1)