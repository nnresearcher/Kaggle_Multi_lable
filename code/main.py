# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:26:20 2018

@author: hasee
"""

import os
from help_function import loss ,accuracy
from help_function import data_generator, createmymodel, str_to_int, modelsave, plotfigure, model_predict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras import layers
from keras.optimizers import SGD,RMSprop
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
from random import random
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
import sys

# para
image_resize = 400 #网络设置的输入图片大小
classes = 228 #最后划分的类别数
batch_size = 4 #每次训练的batch大小
nb_epoch = 1 #训练轮数
dropout = 0.3 #全连接后的dropout比例
lr = 0.000000000001 #学习率初始值
momentum = 0.9 #动量优化
factor = 0.2 #学习速率被降低的因数
patience = 1 #没有进步的训练轮数
min_lr = 0.0000000000000001 #最小学习率
ifenhance = False #训练时是否在线数据增强
ifuseweightold = True #是否使用以前的权重
train_data_hance = False #训练时是否做数据增强

data_index_begin=800000
data_index_end=1000000 # max is 1012449

#8：0.0559 0.5150  0.5 0.52896
#9：loss: 0.0594 - accuracy: 0.4655  0.7 0.5314
version=7
check_point_file_old=r"../data/model/201/densenet_check_point_v" + str(version-1) + ".h5"
check_point_file_new=r"../data/model/201/densenet_check_point_v" + str(version) + ".h5"
submission_csv_path=r"submission169." + str(version) + ".csv"

model_name = r"../data/model/densenet_check_point_v" + str(version) + ".h5"
loss_trend_graph_path = r"../data/model/loss_train_v" + str(version) + ".jpg" #训练阶段的loss趋势图
acc_trend_graph_path = r"../data/model/acc_train_v" + str(version) + ".jpg" #训练阶段的准确率趋势图
model_graph_path = r"../data/model/model.jpg" #模型框架图
model_architecture_path = r"../data/model/model_architecture.json" #模型信息
sample_csv=r'../data/sample_submission.csv'
test_csv_save_path='../data/new_test_data_info.csv'

if os.path.exists(check_point_file_new):
       print("已存在model版本")
       sys.exit(1)

if ((580000<=data_index_end and 580000>=data_index_begin) or(600000<=data_index_end and 600000>=data_index_begin)):
       print("数据区间可能报错")
       #sys.exit(1)

       
target_size=(image_resize,image_resize)
input_shape=(image_resize, image_resize, 3)

print("创建模型")
model = createmymodel(input_shape=input_shape, classes=classes, dropout=dropout, option="DenseNet201")

# 生成测试集和训练集对应的图像路径和标签
print("载入数据")
train_csv_save_path='../data/train_data.csv'
train_data2=pd.read_csv(train_csv_save_path)
train_data=train_data2[:][data_index_begin:data_index_end]
validation_csv_save_path='../data/updata_validation_data.csv'
validation_data2=pd.read_csv(validation_csv_save_path)
validation_data=validation_data2[:][0:1000]
#生成训练数据
print("生成训练数据")
train_data_image_path = list(train_data["image_path"].reset_index(drop=True))
train_data_lable = [eval(x) for x in list(train_data['label'].reset_index(drop=True))]

#生存验证数据validation
validation_data_image_path = list(validation_data["image_path"].reset_index(drop=True))
validation_data_lable = [eval(x) for x in list(validation_data['label'].reset_index(drop=True))]


if ifuseweightold:
    print("载入权重")
    model.load_weights(check_point_file_old)
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
#optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = SGD(lr=lr, momentum=momentum)
model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy])
#生成train_datagen
train_datagen = data_generator( data_image_path=train_data_image_path,
                                data_lable=train_data_lable,
                                batch_size=batch_size,
                                target_size=target_size,
                                reforce=train_data_hance)

#生成validation_datagen
validation_datagen = data_generator( data_image_path=validation_data_image_path,
                                     data_lable=validation_data_lable, 
                                     batch_size=batch_size, 
                                     target_size=target_size, 
                                     reforce=False)

print('Now,we start defining callback functions...')
model_checkpoint = ModelCheckpoint(check_point_file_new, save_best_only=True,
                                   save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)
early_stop = EarlyStopping(monitor='val_loss', patience=3 * patience)
callbacks=[model_checkpoint, reduce_lr, early_stop]

print("Now,we start training...")
history=model.fit_generator(generator=train_datagen,
                            steps_per_epoch= len(train_data_image_path) // batch_size,
                            epochs=nb_epoch,
                            callbacks=callbacks,
                            validation_data=validation_datagen,
                            validation_steps = len(validation_data_image_path) // batch_size,
                            verbose=1)

print("save model...")
modelsave(model, model_architecture_path, model_graph_path)
model.save(check_point_file_new,overwrite=True)
print("predicting")
model_predict(model,image_resize,submission_csv_path,sample_csv,test_csv_save_path)
print("Done...")
