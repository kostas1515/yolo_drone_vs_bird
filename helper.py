import torch
import pandas as pd


def my_collate(batch):
    write=0
    boxes=[]
    for el in batch:
        if write==0:
            pictures=el['image'].unsqueeze(-4)
            write=1
        else:
            pictures=torch.cat((pictures,el['image'].unsqueeze(-4)),0)
        boxes.append(el['bbox_coord'])

    return pictures,boxes

def collapse_boxes(boxes,pw_ph,cx_cy,stride):
    write=0
    mask=[]
    for box in boxes:
        if write==0:
            targets=box
            anchors=torch.stack([pw_ph for p in range(box.shape[0])], dim=0)
            offset=torch.stack([cx_cy for p in range(box.shape[0])], dim=0)
            strd=torch.stack([stride for p in range(box.shape[0])], dim=0)
            write=1
        else:
            targets=torch.cat((targets,box),0)
            
            anchors=torch.cat((anchors,torch.stack([pw_ph for p in range(box.shape[0])], dim=0)),0)
            offset=torch.cat((offset,torch.stack([cx_cy for p in range(box.shape[0])], dim=0)),0)
            strd=torch.cat((strd,torch.stack([stride for p in range(box.shape[0])], dim=0)),0)
        mask.append(box.shape[0])
    return targets,anchors.squeeze(1),offset.squeeze(1),strd.squeeze(1),mask

def expand_predictions(predictions,mask):
    k=0
    write=0
    for i in mask:
        if write==0:
            new=torch.stack([predictions[k,:,:] for p in range(i)], dim=0)
            write=1
        else:
            new=torch.cat((new,torch.stack([predictions[k,:,:] for p in range(i)], dim=0)),0)
        k=k+1
    
    return new


def create_test(x,y,w,h,upper,lower):
    xmin=x.split(';')
    ymin=y.split(';')
    width=w.split(';')
    height=h.split(';')
    x=''
    y=''
    h=''
    w=''
    for i in range(len(width)):
        if(float(width[i])*float(height[i])<=upper)&(float(width[i])*float(height[i])>=lower):
            x=';'.join((xmin[i],x))
            y=';'.join((ymin[i],y))
            h=';'.join((height[i],h))
            w=';'.join((width[i],w))
    return pd.Series([x.rstrip(';'),y.rstrip(';'),w.rstrip(';'),h.rstrip(';')])