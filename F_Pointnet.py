# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:21:44 2020

@author: chuhaofan
"""

from keras import Input, layers
import keras.backend as K
from Modelnew import instance_seg_pointnet,masking_model,T_Net,Amodal_3dbox_estimation_pointnet
from RGBDDataset import NUM_CLASS,NUM_HEADING_BIN,NUM_SIZE_CLUSTER
from Modelnew import mean_size_arr
from keras.models import Model
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, '../sunrgbd'))
from loss_helper import f_pointnet_loss



def create_f_pointnet_body(num_points,NUM_CLASS,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    '''
    '''
    #Input
    input_points = Input(shape = (num_points,6),dtype="float32",name="input_points")
    k =Input(shape=(NUM_CLASS,),dytpe="float32",name="object_class_onehot")
    
    
    #pipeline
    logits,global_feature_expand = instance_seg_pointnet(input_points, k)
    mask,point_cloud_xyz,point_cloud_xyz_stage1,mask_xyz_mean = masking_model(input_points,logits)
    stage1_center=T_Net(point_cloud_xyz_stage1,mask_xyz_mean,k)
    output=Amodal_3dbox_estimation_pointnet(point_cloud_xyz,stage1_center,k,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr)
    return Model(inputs=input_points,outputs=[logits,stage1_center,output])



def decode_scores(args,NUM_CLASS,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    stage1_center , output = args
    center_mid = output[:,0:3]
    center = center_mid + stage1_center # Bx3
    
    heading_scores = output[:,3:3+NUM_HEADING_BIN] #BxNH
    heading_residuals_normalized = output[:,3+NUM_HEADING_BIN:3+NUM_HEADING_BIN*2] #BXNH
    heading_residuals = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN)
    
    size_scores = output[:,3+NUM_HEADING_BIN*2:3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER]
    size_residual_normalize = output[:,3+NUM_HEADING_BIN*2:3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER: 3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4]    
    size_residuals_normalized = K.reshape(size_residual_normalize, (-1, NUM_SIZE_CLUSTER, 3))
    size_residuals = size_residuals_normalized * K.expand_dims(K.constant(mean_size_arr, dtype=K.float32), 0)
    return [center,heading_scores,heading_residuals_normalized,heading_residuals,size_scores,size_residuals_normalized,size_residuals]


def create_f_pointnet(num_points,):
    #inputs
    mask_label = Input((num_points,))
    center_label =Input((3,))
    heading_class_label = Input((NUM_HEADING_BIN,))
    heading_residual_label = Input((NUM_HEADING_BIN,))
    size_class_label = Input((NUM_SIZE_CLUSTER,))
    size_residual_label = Input((NUM_SIZE_CLUSTER,3))
    #main  pipeline 
    f_pointnet_body = create_f_pointnet_body(num_points,NUM_CLASS,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr)
    logits,stage1_center,output = f_pointnet_body.outputs
    
    decoded_outputlayer= layers.Lambda(decode_scores, 
                                    arguments={'num_class':NUM_CLASS, 
                                           'num_heading_bin':NUM_HEADING_BIN,
                                           'num_size_cluster':NUM_SIZE_CLUSTER,
                                           'mean_size_arr':mean_size_arr},
                                    name='decore_scores')([stage1_center , output])
    center,heading_scores,heading_residuals_normalized,heading_residuals,\
        size_scores,size_residuals_normalized,size_residuals =decoded_outputlayer
    '''
    Label:
        0 - mask_label,
        1 - center_label, 
        2 - heading_class_label,
        3 - heading_residual_label, 
        4 - size_class_label,
        5 - size_residual_label,
    Predictions:
        6 - logits
        7 - stage1_center,
        8 - center, 
        9 - heading_scores,        
        10 - heading_residuals_normalized,
        11 - heading_residuals
        12 - size_scores,
        13 - size_residuals_normalized,
        14 - size_residuals
    '''
    args_loss = [mask_label,
                 center_label,
                 heading_class_label,
                 heading_residual_label,
                 size_class_label,
                 size_residual_label,
                 logits,
                 stage1_center,
                 center,
                 heading_scores,
                 heading_residuals_normalized,
                 heading_residuals,
                 size_scores,
                 size_residuals_normalized,
                 size_residuals]   
    #generate the loss function to the Lambda Layer
    loss = layers.Lambda(f_pointnet_loss,output_shape=(9,),name="f_pointnet_loss")(args_loss)
    return Model(inputs=[*f_pointnet_body.inputs,
                         mask_label,
                         center_label,
                         heading_class_label,
                         heading_residual_label,
                         size_class_label,
                         size_residual_label],outputs=loss,name='f_pointnet_net')
