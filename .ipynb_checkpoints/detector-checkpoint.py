from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys



df = pd.read_csv('../annotations.csv')

model = Darknet("../cfg/yolov3.cfg")
model.load_weights("../yolov3.weights")


optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer.zero_grad()

batch=1

epochs=20

for e in range(epochs):
    prg_counter=0
    print("epoch "+str(e))
    for index, row in df.iterrows():
        imgpath='../images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        inp = get_test_input(imgpath)
        targets=torch.tensor([[[row['x']*(544/1980),row['y']*(544/1080),row['width']*(544/1980),row['height']*(544/1080),1,1]]])
        pred = model(inp, torch.cuda.is_available())
        loss=util.yolo_loss(pred,targets)
        loss.backward()
        optimizer.step()
        sys.stdout.write('\rProgress is ' +str(prg_counter/8771*100)+'%' ' loss is: '+ str(loss.item()))
        sys.stdout.flush()
        time.sleep(1)
        prg_counter=prg_counter+1
    
#     index=1
#     imgpath='../images/'+df['filename'][index]+'_img'+df['framespan'][index].split(':')[0]+'.jpg'
#     inp = get_test_input(imgpath)
#     targets=torch.tensor([[[df['x'][index]*(416/1980),df['y'][index]*(416/1080),df['width'][index]*(416/1980),df['height'][index]*(416/1080),1,1]]])
#     pred = model(inp, torch.cuda.is_available())
#     loss=util.yolo_loss(pred,targets)
#     print(loss)
#     loss.backward()
#     optimizer.step()
    
#     cost=cost+loss
#     if(k==batch):
#         print(cost)
#         cost.backward()
#         optimizer.step()
#         cost=0
#         k=0
#     k=k+1

