# -*- coding: utf-8 -*-
"""
Created on Thu May  3 23:01:31 2018

@author: hasee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import json
from tqdm import tqdm 
import numpy as np
import io
import os
import sys
import json
import urllib3
import multiprocessing 
from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.models import Model
from densenet_updated import DenseNet121,DenseNet169,DenseNet201
import random
from math import floor
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import json



sample_csv=r'../data/sample_submission.csv'
test_json=r'../data/test.json'
train_json=r'../data/train.json'
validation_json=r'../data/validation.json'

def modelpredict(imagepath, model, resize):
    image = Image.open(imagepath)
    image = image.resize(resize, Image.ANTIALIAS)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = np.expand_dims(image, axis=0)
    prop = model.predict(image)
    return prop

    
def plotfigure(info, history, savepath):
    assert info in ["accuracy","loss"], "info should be in ['accuracy', 'loss']"
    fig = plt.figure(1)
    if info == "accuracy":
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("accuracy")
    elif info == "loss":
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc="upper left")
    plt.savefig(savepath)
    plt.close(1)


def modelsave(model, modelpath, modelgraphpath, withweight=False, plotmodel=False):
    if withweight == False:
        json_string = model.to_json()
        open(modelpath,'w').write(json_string)
    else:
        model.save(modelpath)
    if plotmodel == True:
        assert modelgraphpath is not None, "modelgraphpath shoule be specified if plotmodel is True"
        plot_model(model, show_shapes=True, to_file=modelgraphpath)

def str_to_int(str_data):
       temp=[]

       for i in range(len(str_data)):
              for ii in range(1,len(str_data[i])):
                     if (str_data[i][ii]!=',') & (str_data[i][ii]!=' ') & (str_data[i][ii]!='[') & (str_data[i][ii]!=']'):
                            temp.append(int(str_data[i][ii]))
              str_data[i]=temp
       return str_data


def createmymodel(input_shape, classes, dropout, option="DenseNet201"):
    assert option in ["DenseNet121", 'DenseNet121_scratch','DenseNet169','DenseNet201'], \
              "the option should be in ['Densenet121', 'DenseNet121_scratch']"
    if option == "DenseNet121_scratch":
        basemodel = DenseNet121(include_top=False,
                         input_tensor=None,
                         input_shape=input_shape,
                         pooling=None)
    elif option == "DenseNet121":
        basemodel = DenseNet121(include_top=False,
                         weights="imagenet",
                         input_tensor=None,
                         input_shape=input_shape,
                         pooling=None)
    elif option == "DenseNet169":
        basemodel = DenseNet169(include_top=False,
                                weights='imagenet',
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None)
    elif option == "DenseNet201":
        basemodel = DenseNet201(include_top=False,
                                weights='imagenet',
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None)
    x = basemodel.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(2048, activation='relu')(x)
    
    predicts = Dense(classes, activation='sigmoid')(x)
    model = Model(input=basemodel.input, outputs=predicts)
    return model

def get_csv_info(csv_path):
    label_csv=pd.read_csv(csv_path)
    return label_csv
    
def get_max_and_min_index():
       file = open(train_json,'r',encoding='utf-8')  
       s = json.load(file)
       annotations=s['annotations']
       max_index=-1
       min_index=10000
       for annotation in annotations:
              for i in range(len(annotation['labelId'])):
                     if int(annotation['labelId'][i])>max_index:
                            max_index=int(annotation['labelId'][i])
                     if int(annotation['labelId'][i])<min_index:
                            min_index=int(annotation['labelId'][i])
       return max_index,min_index

def change_label(annotation_label):
       # annotation_label :  annotation['labelId']
       new_label=[]
       for i in range(0,228):
              new_label.append(0)
       for i in range(len(annotation_label)):
              index = int(annotation_label[i])
              new_label[index-1]=1
       return new_label
       
def change_Id(annotation_imageId,type_data):
       
       annotation_imageId=".".join([annotation_imageId, 'jpg'])
       if type_data=='train':
              IMAGE_save_path="/".join(['../data/train', annotation_imageId])
       if type_data=='validation':
              IMAGE_save_path="/".join(['../data/validation', annotation_imageId])
       if type_data=='test':
              IMAGE_save_path="/".join(['../data/test', annotation_imageId])
       return IMAGE_save_path
       
def get_new_dict(annotations,data_type):
       for annotation in annotations:
              if data_type !='test':
                     annotation['labelId']=change_label(annotation['labelId'])
                     annotation['imageId']=change_Id(annotation['imageId'],data_type)
              else:
                     annotation['imageId']=change_Id(annotation['imageId'],data_type)
              
       return annotations
       
       
       
def get_image_path_and_label(annotations):
       image_path=[]
       image_label=[]
       for annotation in annotations:
              image_path.append(annotation['imageId'])
              image_label.append(annotation['labelId'])                    
       return image_path,image_label
       
       
def get_new_train_csv(save_path):
       file = open(train_json,'r',encoding='utf-8')  
       s = json.load(file) 
       annotations=s['annotations']
       annotations=get_new_dict(annotations,'train')
       image_path,image_label=get_image_path_and_label(annotations)
       save_csv = pd.DataFrame({"image_path":image_path, "label":image_label})
       save_csv.to_csv(save_path, index=False)
       
def get_new_validation_csv(save_path):
       file = open(validation_json,'r',encoding='utf-8')  
       s = json.load(file) 
       annotations=s['annotations']
       annotations=get_new_dict(annotations,'validation')
       
       image_path,image_label=get_image_path_and_label(annotations)
       save_csv = pd.DataFrame({"image_path":image_path, "label":image_label})
       save_csv.to_csv(save_path, index=False)
       
def get_new_test_csv(save_path):
       file = open(test_json,'r',encoding='utf-8')  
       s = json.load(file) 
       ss=s['images']
       image_path=[]
       for i in range(len(ss)):
              path=".".join([str(i+1),'jpg'])
              paths="/".join(['../data/test', path])
              image_path.append(paths)
       save_csv = pd.DataFrame({"image_path":image_path})
       save_csv.to_csv(save_path, index=False)


def loss(y_true, y_pred):
    y_true_valid = y_true
    y_pred_valid = y_pred
    return K.mean(K.binary_crossentropy(y_true_valid, y_pred_valid), axis=-1)


def accuracy(y_true, y_pred):
       
    valid_poss = K.cast(y_pred > 0.5, dtype=K.floatx())
    valid = valid_poss*y_true
    a=K.sum(valid)
    valid_posss=valid_poss+y_true
    valid_possss = K.cast(valid_posss > 0.5, dtype=K.floatx())
    b=K.sum(valid_possss)
    #y_pred_valid = K.cast(y_pred_valid > 0.5, dtype=K.floatx())
    #return K.mean(K.cast(K.equal(y_true_valid,y_pred_valid),K.floatx()))
    return 2*a/(b+a)


def data_generator(data_image_path, data_lable, batch_size, target_size, reforce=False):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    while True:
        Input = []
        output = []
        for _ in range(batch_size):
            random_image_index= random.randint(0, len(data_image_path)-1)
            image = Image.open(data_image_path[random_image_index])
            image = image.resize(target_size, Image.ANTIALIAS)
            image = np.array(image)
            Input.append(image)
            output.append(data_lable[random_image_index])
        Input = np.asarray(Input)
        output = np.asarray(output)
        Input = Input.astype('float32')
        Input /= 255
        yield (Input, output)
        
        
        
def model_predict(model,image_resize,submission_csv_path,sample_csv,test_csv_save_path):


       target_size=(image_resize, image_resize)
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
       

           
def download_data(train_data_download_complete,test_data_download_complete,validation_data_download_complete):
       test_json=r'../data/test.json'
       train_json=r'../data/train.json'
       validation_json=r'../data/validation.json'
       if train_data_download_complete!=True:
              download_pic(train_json, r'../data/train')
       if test_data_download_complete!=True:
              download_pic(test_json, r'../data/test')
       if validation_data_download_complete!=True:
              download_pic(validation_json, r'../data/validation')