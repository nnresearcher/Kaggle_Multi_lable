# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:45:52 2018

@author: hasee
"""
import pandas as pd
import os 
from tqdm import tqdm
train_csv_save_path='../data/new_train_data_info.csv'
train_data=pd.read_csv(train_csv_save_path)

drop_index=[]
for i in tqdm(range(0,100000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/1/")

       

for i in tqdm(range(100000,200000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/2/")


for i in tqdm(range(200000,300000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/3/")


for i in tqdm(range(300000,400000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/4/")


for i in tqdm(range(400000,500000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/5/")


for i in tqdm(range(500000,600000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/6/")


for i in tqdm(range(600000,700000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/7/")


for i in tqdm(range(700000,800000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/8/")

for i in tqdm(range(800000,900000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/9/")


for i in tqdm(range(900000,1000000)):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/10/")



for i in tqdm(range(1000000,len(train_data['image_path']))):
       train_data['image_path'][i]=train_data['image_path'][i].replace("../data/train/","../data/train/11/")



for i in tqdm(range(len(train_data['image_path']))):
       if (os.path.exists(train_data['image_path'][i])==False):
              drop_index.append(i)     
              

train_data=train_data.drop(drop_index)
train_data.to_csv('../data/train_data_rename.csv', index=False)






























