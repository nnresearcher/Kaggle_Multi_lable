# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:45:52 2018

@author: hasee
"""



from help_function import createmymodel, modelpredict


classes = 228 #最后划分的类别数
image_resize = 400 #网络设置的输入图片大小
dropout = 0.0 #全连接后的dropout比例
target_size=(image_resize, image_resize)
input_shape=(image_resize, image_resize, 3)



print('creat model')
model = createmymodel(input_shape=input_shape, classes=classes, dropout=dropout, option="DenseNet201")


print('load model para')
version=5
check_point_file = r"../data/model/201/densenet_check_point_v" + str(version) + ".h5"
submission_csv_path=r"submission1." + str(version) + ".csv"
model.load_weights(check_point_file)


import pandas as pd
from tqdm import tqdm
import json
sample_csv=r'../data/sample_submission.csv'
test_json=r'../data/test.json'
train_json=r'../data/train.json'
validation_json=r'../data/validation.json'  

test_csv_save_path='../data/new_test_data_info.csv'
rankdata = pd.read_csv(test_csv_save_path, header=None)
data_index=rankdata[0]
pre_result=[]
for i in tqdm(range(len(data_index)-1)):
       ii=i+1
       index=data_index[ii]
       prop=[]
       prop = modelpredict(index, model, target_size)
       result_list=[]
       for iii in range(len(prop[0])):
              if prop[0][iii]>0.5:
                     result_list.append(iii+1)
       pre_result.append(result_list)
              
save_csv = pd.DataFrame({"image_path":pre_result})
save_csv.to_csv('test11.csv', index=False)
testtt=pd.read_csv('test11.csv')
b=[]
for info in testtt['image_path']:
       a=info.replace(',','').replace('[','').replace(']','')
       b.append(a)

train_data2=pd.read_csv(sample_csv)
d=train_data2['image_id']
save_csv = pd.DataFrame({"image_id":d,"label_id":b})
save_csv.to_csv(submission_csv_path, index=False)