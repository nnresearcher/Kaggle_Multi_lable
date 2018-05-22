# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:05:30 2018

@author: hasee
"""

import os.path
import shutil

      
for i in range(900000,1000000):
       index=i+1
       pic_path=r"../data/train/" + str(index) + ".jpg"
       save_path=r"../data/train/10"
       pic_path2=r"../data/train/train2/" + str(index) + ".jpg"
       debug_path=r"../data/train/10/"+ str(index) + ".jpg"
       if (os.path.exists(pic_path) or os.path.exists(pic_path2)):
              if (os.path.exists(debug_path)==False):
                     if os.path.exists(pic_path):
                            shutil.move(pic_path,save_path)
              
                     else:
                            shutil.move(pic_path2,save_path)
for i in range(1000000,1100000):
       index=i+1
       pic_path=r"../data/train/" + str(index) + ".jpg"
       save_path=r"../data/train/11"
       pic_path2=r"../data/train/train2/" + str(index) + ".jpg"
       debug_path=r"../data/train/11/"+ str(index) + ".jpg"
       if (os.path.exists(pic_path) or os.path.exists(pic_path2)):
              if (os.path.exists(debug_path)==False):
                     if os.path.exists(pic_path):
                            shutil.move(pic_path,save_path)
              
                     else:
                            shutil.move(pic_path2,save_path)