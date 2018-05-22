# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:45:52 2018

@author: hasee
"""


from PIL import Image
import imghdr
import pandas as pd
from tqdm import tqdm
train_csv_save_path='../data/train_data_rename.csv'
train_data=pd.read_csv(train_csv_save_path)
drop_index=[]
for i in tqdm(range(580000,600000)):
       path=train_data['image_path'][i]
       if (imghdr.what(path)==False):
              drop_index.append(i)
              print(i)
train_data=train_data.drop(drop_index)
#a=Image.open(path) 		

if imghdr.what(path):
       print(1)
else:
       print(0)
train_data.to_csv('../data/train_data.csv', index=False)


import os
for i in tqdm(range(580000,600000)):
       if (os.path.exists(train_data['image_path'][i])==False):
              drop_index.append(i)     
              

train_data=train_data.drop(drop_index)
train_data.to_csv('../data/train_data_rename.csv', index=False)