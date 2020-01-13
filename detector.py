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

'''
when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
if you start fresh then first model.load_weights and then make it parallel
'''
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(net)

    model.to(device)
    
    PATH = './darknet.pth'
    weights = torch.load(PATH)
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
for e in range(epochs):
    prg_counter=0
    total_loss=0
    print("\n epoch "+str(e))
    for index, row in df.iterrows():
        optimizer.zero_grad()
        imgpath='../images/images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        inp = get_test_input(imgpath)
        targets=torch.tensor([[[row['x']*(416/1980),row['y']*(416/1080),row['width']*(416/1980),row['height']*(416/1080),1,1]]])
        pred = model(inp, torch.cuda.is_available())
        pred=pred.to(device='cuda')
        targets=targets.to(device='cuda')
        loss=util.yolo_loss(pred,targets)
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r Progress is ' +str(prg_counter/9570*100)+'%' ' loss is: '+ str(loss.item()))
        sys.stdout.flush()
        prg_counter=prg_counter+1
        total_loss=total_loss+loss.item()
    torch.save(model.state_dict(), PATH)
    print('\n total average loss is '+str(total_loss/9570))