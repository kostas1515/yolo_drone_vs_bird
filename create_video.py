import cv2
import os
import pandas as pd

image_folder = '../anottated_test_images'
video_name = 'detection.avi'

df = pd.read_csv('../test+pred_annotations.csv')

row=df.iloc[0]
image_path='../anottated_test_images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
frame = cv2.imread(image_path)
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter(video_name,fourcc, 25, (width,height))


for index, row in df.iterrows():
    if (row['obj_id']==1):
        image_path='../anottated_test_images/'+row['filename']+'_img'+row['framespan'].split(':')[0]+'.jpg'
        video.write(cv2.imread(image_path))




cv2.destroyAllWindows()
video.release()