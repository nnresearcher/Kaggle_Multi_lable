# -*- coding: utf-8 -*-
"""
Created on Fri May  4 00:48:10 2018

@author: hasee
"""

##download train pic ####
import json
from urllib.request import urlretrieve  
from tqdm import tqdm 
def urllib_download(IMAGE_URL,IMAGE_save_path):   
    urlretrieve(IMAGE_URL, IMAGE_save_path) 

train_json=r'../data/train.json'
file = open(train_json,'r',encoding='utf-8')  
s = json.load(file) 
ss=s['images']
ccc=ss[110:]
err_id=[]
for sss in tqdm(ccc):
       IMAGE_URL=sss['url']
       IMAGE_id=sss['imageId']
       IMAGE_id_jpg=".".join([IMAGE_id, 'jpg'])
       IMAGE_save_path="/".join(['../data/train', IMAGE_id_jpg])
       try:
              urllib_download(IMAGE_URL,IMAGE_save_path) 
              #print(IMAGE_id+".jpg download over")
       except:
              err_id.append(IMAGE_id)
              #print(IMAGE_id+'.jpg counld not download')
       