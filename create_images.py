import numpy as np
import pandas as pd
import cv2
import math 
from PIL import Image, ImageDraw 
import time

import psutil

df = pd.read_csv('../test+pred_annotations.csv')
for index, row in df.iterrows():
    if ((row['obj_id']==0)&(row['filename']=='gopro_001')):
        image_path='../anottated_test_images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        image = Image.open(image_path)
#     else:
#         image_path='../test_images/'+row['filename']+'_'+str(row['obj_id'])+'_img'+row['framespan'].split(':')[0]+'.jpg'
#         image = Image.open(image_path)
    
        im_width, im_height = image.size
        x=row['x']
        y=row['y']
        w=row['width']
        h=row['height']
        #     print(x,y,w,h)

        xmin = x
        ymin = y 
        xmax = (x + w)
        ymax = (y + h)

        #     print(xmin,ymin,xmax,ymax)
        (left, right, top, bottom) = (xmin, xmax,
                      ymin, ymax)
        image1=ImageDraw.Draw(image)
        image1.rectangle([left,top,right,bottom], outline ="#0cf573", width=4)
        image1.text([left,top],text='GT',outline ="#0cf573")
    
#     xmin = row['pred_xmin']
#     ymin = row['pred_ymin']
#     xmax = row['pred_xmax']
#     ymax = row['pred_ymax']
    
#     (left, right, top, bottom) = (xmin, xmax,
#                   ymin, ymax)
    
#     image1.rectangle([left,top,right,bottom], outline ="#ff8800", width=4)
#     image1.text([left,top],text=str(row['iou']),outline ="#ff8800")
    
        image.save('../anottated_test_images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg')