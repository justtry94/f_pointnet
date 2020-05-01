# -*- coding: utf-8 -*-
"""
Created on 2020.03.08

@author: Chuhao Fan 
"""
import numpy as np
import tensorflow as tf
from keras import Input
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout, Layer, Lambda,subtract
from keras.layers import BatchNormalization, Conv2D, ReLU, \
                            GlobalMaxPool2D, MaxPool2D, Concatenate
from keras import regularizers
import keras.backend as K
log_dir = 'logs/000/'
end_points = {}
type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
class2type = {type2class[t]:t for t in type2class}
type2onehotclass={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
type_mean_size = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                  'bed': np.array([2.114256,1.620300,0.927272]),
                  'bookshelf': np.array([0.404671,1.071108,1.688889]),
                  'chair': np.array([0.591958,0.552978,0.827272]),
                  'desk': np.array([0.695190,1.346299,0.736364]),
                  'dresser': np.array([0.528526,1.002642,1.172878]),
                  'night_stand': np.array([0.500618,0.632163,0.683424]),
                  'sofa': np.array([0.923508,1.867419,0.845495]),
                  'table': np.array([0.791118,1.279516,0.718182]),
                  'toilet': np.array([0.699104,0.454178,0.756250])}
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_CLASS = 10
mean_size_arr = mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))#(NS,3)
for i in range(NUM_SIZE_CLUSTER):
    mean_size_arr[i,:] = type_mean_size[class2type[i]]#按物体类别排序的anchorbox 大小


def mlp(x, num_output, batch_norm = True,activ='relu'):
    x = Conv2D(num_output, [1,1], activation=activ)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x
def exp_dim(tensor, num_expand_axis):
    return K.expand_dims(tensor,num_expand_axis)
def concat_dim(part1, part2,axi):
    return K.concatenate([part1,part2], axis=axi)
def tile_dim(global_feature, dims):
    return K.tile(global_feature, dims)
def squeeze(inputs, squeeze_aixs):
    return K.squeeze(inputs,squeeze_aixs)

def slice_keras(inputs,begin,size):
    return K.slice(inputs,begin,size)
def masking(inputs):
    mask = K.slice(inputs,[0,0,0],[-1,-1,1]) < K.slice(inputs,[0,0,1],[-1,-1,1]) 
    #compare two element which means possbility of object or clutter  of chanenal 3 
    mask = K.cast(mask,"float32")
    #return mask
    return mask
    #mask = tf.to_float(mask)
'''
a= masking(np.random.random((3,3,2)))
inter = K.tile(a,[1,1,3])
print(inter)
'''
def mask_count(x,axis=1):
    mask_count1 = K.sum(x,axis,keepdims=True)
    # BxNx1 to Bx1x1 calculate the true number (object)
    mask_count2 = K.tile(mask_count1 ,[1,1,3])
    #mask_count2 = tf.convert_to_tensor(mask_count2)
    return mask_count2
    #Bx1x1  to BX1X3 useful to later x y z mean  calculate 
    #mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True), [1,1,3]) # Bx1x3
def xyz_mean(point_cloud_xyzs,masked,mask_counts,num_point=2048):
    
    inter = K.tile(masked, [1,1,3])
    inter = inter*point_cloud_xyzs#BxNx(1*3) *BxNx3
    mask_xyz_mean_sum = K.sum(inter, axis=1, keepdims=True)#Bx1x3
    mask_xyz_mean = mask_xyz_mean_sum/K.maximum(mask_counts,1)#Bx1x3
    #mask_xyz_mean_expand = K.tile(mask_xyz_mean, [1,num_point,1])
    #point_cloud_xyz_stage1 = point_cloud_xyzs - mask_xyz_mean_expand#BxNX3
    
    return  mask_xyz_mean#,mask_xyz_mean_expand

def elementMut(a,b):
    return a*b

def center_caculate(a,b):
    stage1_center = a + K.squeeze(b, axis=1)
    end_points['stage1_center'] = stage1_center
    return stage1_center

def submean(a,b):
    c = a - K.expand_dims(b, 1)
    return c

def center_calculate(inputs1,inputs2,begin,size):    
    center = K.slice(inputs1,begin,size )#[0,0], [-1,3]
    center = center + inputs2 # Bx3#??????????????????????????
    end_points['center'] = center
    return center   
def reshape_keras(inputs,shape):
    return K.reshape(inputs, shape) 

def loding_heading_scores(heading_scores):##Keras tensor?????? cab be written  
    end_points['heading_scores'] = heading_scores
    
def loding_heading_load_residuals_normalized(heading_residuals_normalized):
    end_points['heading_residuals_normalized'] = heading_residuals_normalized
    
def loding_heading_load_residuals(heading_residuals_normalized):
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN)
def heading_load_residuals(heading_residuals_normalized):
    heading_residuals = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN)
    return heading_residuals   
def loding_size_scores(size_scores):
    end_points['size_scores'] = size_scores
    
def loding_size_residuals_normalized(size_residuals_normalized):
    end_points['size_residuals_normalized'] = size_residuals_normalized
    
def loding_size_residuals(size_residuals_normalized,mean_size_arr):
    end_points['size_residuals'] = size_residuals_normalized * K.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0)
def size_residuales(size_residuals_normalized,mean_size_arr):
    size_residuals = size_residuals_normalized * K.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0) 
    return size_residuals
def instance_seg_pointnet(input_points,k):
    
    #batch_size = input_points.get_shape()[0].value#B
    # print(batch_size)
    N = input_points.get_shape()[1].value#N
    print(N)
    #end_points = {}
    input_points = Reshape((N,6,1))(input_points)
    #input_points = Lambda(exp_dim, arguments={'num_expand_axis': -1})(input_points) #BxNx6   to   BxNx6x1
    #K.expand_dims(input_points, -1)#BxNx6   to   BxNx6x1
    #input_points = Input(shape=(num_points, 3))
    x = input_points
    x = Conv2D(64, [1,6], activation='relu')(x)
    x = BatchNormalization()(x)
    #get point  local feature 
    seg_part1 = mlp(x,64,batch_norm = True,activ='relu')
    x = mlp(seg_part1,64,batch_norm = True,activ='relu')
    x = mlp(x,128,batch_norm = True,activ='relu')
    x = mlp(x,1024,batch_norm = True,activ='relu')
    # get global feature 
    global_feature= GlobalMaxPool2D()(x)#Bx1x1024
    global_feature = Reshape((1,1,1024))(global_feature)
    #feature concatenate from point feature ,sematic feature and global feature
    
    one_hot_vec_modified1 = Lambda(exp_dim, arguments={'num_expand_axis': 1})(k)# BxK to Bx1xK 
    # which form the input k should be  ???????????????????????
    one_hot_vec_modified = Lambda(exp_dim, arguments={'num_expand_axis': 1})(one_hot_vec_modified1)# Bx1xK to Bx1x1xK
    #one_hot_vec_modified1 = K.expand_dims(k,1)# BxK to Bx1xK
    #one_hot_vec_modified = K.expand_dims(one_hot_vec_modified1,1)# Bx1xK to Bx1x1xK
    global_feature = Lambda(concat_dim,arguments={"part2":one_hot_vec_modified,"axi":3},name="first_concate")(global_feature)
    #global_feature = K.concat([global_feature,one_hot_vec_modified], axis=3) #BX1X1X1024 + Bx1x1xK  =Bx1x1x(1024+K)
    #BX1X1X1024 + Bx1x1xK  =Bx1x1x(1024+K)

    global_feature_expand = Lambda(tile_dim,arguments={"dims": [1,N,1,1]},name="expand")(global_feature)
    #global_feature_expand = K.tile(global_feature, [1, N, 1, 1])
    #Bx1x1x(1024+K)  to BxNx1x(1024+K)
    x = Lambda(concat_dim,arguments={'part2':global_feature_expand,'axi':3},name="feature_concate")(seg_part1)
    #x = K.concat([seg_part1, global_feat_expand],axis=3) 
    #BXNX1X(1024+K) + BxNx1x64  =BxNx1x(1088+K)
    #keras结构最小的操作单元是layer 而不是tensor
    
    x = mlp(x,512,batch_norm = True,activ='relu')#BxNx1x512
    x = mlp(x,256,batch_norm = True,activ='relu')#BxNx1x256
    x = mlp(x,128,batch_norm = True,activ='relu')#BxNx1x128
    x = Dropout(0.5, noise_shape=None, seed=None)(x)
    logits = mlp(x,2,batch_norm = True,activ=None)#BxNx1x2
    logits = Lambda(squeeze,arguments={"squeeze_aixs": 2},name="squeeze_axis")(logits) #BxNx2 降维度
    #logits = tf.squeeze(logits, [2])
    return logits,global_feature_expand


def masking_model(input_points,logits):
    N = input_points.get_shape()[1].value
    
    ###build the mask and caculate the mask number 
    mask = Lambda(masking,name="masking_process")(logits)#matrix full of boolean value
    mask_counted = Lambda(mask_count,name="seg_point_count")(mask)  
    ##get the segmented object points 
    input_points = Reshape((2048,6))(input_points)
    point_cloud_xyz = Lambda(slice_keras,arguments={"begin" : [0,0,0], "size": [-1,-1,3]},name="catch_coordinate")(input_points)
    
    #point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3
    
    # ---- Subtract points mean ---- Fig4 (3) normalize  to get the mask center then get point coordinate based on mask origin 
    mask_xyz_mean = Lambda(xyz_mean,arguments={"masked":mask,"mask_counts":mask_counted},name="substract_points_mean")(point_cloud_xyz)
    mask_xyz_mean_expand = Lambda(tile_dim, arguments={"dims":[1,N,1]})(mask_xyz_mean)
    point_cloud_xyz_stage1 =subtract([point_cloud_xyz, mask_xyz_mean_expand])
    return mask,point_cloud_xyz,point_cloud_xyz_stage1,mask_xyz_mean
    #How to set the Input layer   
def T_Net(point_cloud_xyz_stage1,mask_xyz_mean,k):
    # ---- Regress 1st stage center ----  T-Net  for estimating residual center 
    net = Lambda(exp_dim,arguments={'num_expand_axis': 2})(point_cloud_xyz_stage1)#BxNx3 to BxNx1x3
    net = mlp(net,128,batch_norm = True,activ='relu')#BxNx1x3 to BxNx1x128
    net = mlp(net,256,batch_norm = True,activ='relu')#BxNx1x128 to BxNx1x128
    net = mlp(net,512,batch_norm = True,activ='relu')#BxNx1x128 to BxNx1x256
    #mask_expand_inter = Lambda(exp_dim,arguments={'num_expand_axis': -1})(mask)#BxNx1 to BxNx1x1  
    #mask_expand = Lambda(tile_dim,arguments={"dims": [1,1,1,256]})(mask_expand_inter)#BxNx1x1 to  BxNx1x256
    #masked_net = Lambda(elementMut,arguments={"b": mask_expand})(net)#BxNx1x256 * BxNx1x256 = BxNx1x256
    net = GlobalMaxPool2D()(net)#Bx1x256
    net = Reshape((1,1,512))(net)
    net = Reshape((512,))(net)
    
   #net = Lambda(squeeze,arguments={"squeeze_aixs":[1,2]})(net)#1x1x256 to Bx256
    global_feature_tnet = Lambda(concat_dim,arguments={'part2':k,'axi':1},name="fature_concate")(net)
    #global_feature_tnet = Lambda(concat_dim,arguments={'part2':k, "axi"：1})(net)##Bx(512+K)
    net = Dense(256, activation='relu')(global_feature_tnet)#Bx256
    net = Dense(128, activation='relu')(net)##Bx128
    stage1_center_inter = Dense(3, activation=None)(net)#BX3
    #------ Fig4 (3)---- estimate true center of the complete object translation
    stage1_center = Lambda(center_caculate ,arguments={"b": mask_xyz_mean},name="center_estimation")(stage1_center_inter)
    # Bx3 +(Bx1x3) get 1 out
    return stage1_center

def Amodal_3dbox_estimation_pointnet(point_cloud_xyz,stage1_center,k,NUM_HEADING_BIN,NUM_SIZE_CLUSTER):
    batchsize = point_cloud_xyz.get_shape()[0].value
    # ---- Subtract stage1 center ----v1 Amodal 3d box Estimation PointNet 
    point_cloud_xyz_submean = Lambda(submean,arguments={"b":point_cloud_xyz } )(stage1_center)#BxNx3
    #net =  Lambda(exp_dim,arguments={'num_expand_axis': 2})(point_cloud_xyz_submean)
    net = Reshape((2048,1,3))(point_cloud_xyz_submean)
    net = mlp(net,128,batch_norm = True,activ='relu')
    net = mlp(net,128,batch_norm = True,activ='relu')
    net = mlp(net,256,batch_norm = True,activ='relu')
    net = mlp(net,512,batch_norm = True,activ='relu')
 
    #3dbox_mask_expand_inter= Lambda(exp_dim,arguments={'num_expand_axis': -1})(mask)#BxNx1 to BxNx1x1     
    #3dbox_mask_expand_inter = Lambda(exp_dim,arguments={'num_expand_axis': -1})(mask)##BxNx1 to BxNx1x1
    #3dbox_mask_expand = Lambda(tile_dim,arguments={"dims": [1,1,1,512]})(3dbox_mask_expand_inter)#BxNx1x1 to  BxNx1x256
    #3dbox_masked_net = Lambda(elementMut,arguments={"b": 3dbox_mask_expand})(net)#BxNx1x256 * BxNx1x256 = BxNx1x256
    net = GlobalMaxPool2D()(net)
    net = Reshape((512,))(net)
    global_feature_3dbox = Lambda(concat_dim,arguments={"part2" :k,"axi" :1})(net)
    net = Dense(256, activation='relu')(global_feature_3dbox)#Bx256
    net = Dense(128, activation='relu')(net)##Bx128
    output = Dense(3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation=None)(net)#Bx3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
    '''
    # ---- use slice function to separte the result ----
    #center calculate  
    center = Lambda(center_calculate,arguments={"inputs2":stage1_center,"begin":[0,0],"size": [-1,3]},name="get_center")(output)
    
    #heading calculate
    heading_scores = Lambda(slice_keras,arguments={"begin":[0,3],"size": [-1,NUM_HEADING_BIN]},name="get_heading_scores")(output)
    heading_residuals_normalized = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN],"size": [-1,NUM_HEADING_BIN]},name="get_heading_residuals_normalized")(output)
    heading_residuals = Lambda(heading_load_residuals,name="get_heading_residual")(heading_residuals_normalized)
    
    #size calculate 
    size_scores = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN*2],"size": [-1,NUM_SIZE_CLUSTER]},name="get_size_scores")(output)
    size_residuals_normalize = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER],"size": [-1,NUM_SIZE_CLUSTER*3]},name="size_residuals_normalize")(output)
    size_residuals_normalized = Reshape((NUM_SIZE_CLUSTER,3))(size_residuals_normalize)
    #size_residuals_normalized = Lambda(reshape_keras,arguments={"shape":(batchsize, NUM_SIZE_CLUSTER,3)})(size_residuals_normalize)
    size_residuals = Lambda(size_residuales,arguments={"mean_size_arr" : mean_size_arr},name="get_size_residuals")(size_residuals_normalized)
    outputs = [center,heading_scores,heading_residuals_normalized,heading_residuals,size_scores,size_residuals_normalized,size_residuals]
    '''
    return output #return outputs

def decode_scores(args,NUM_CLASS,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    stage1_center , output = args
    center_mid = output[:,0:3]
    center = center_mid + stage1_center # Bx3
    
    heading_scores = output[:,3:3+NUM_HEADING_BIN] #BxNH
    heading_residual_normalized = output[:,3+NUM_HEADING_BIN:3+NUM_HEADING_BIN*2] #BXNH
    heading_residuals = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN)
    
    size_score = output[:,3+NUM_HEADING_BIN*2:3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER]
    size_residual_normalize = output[:,3+NUM_HEADING_BIN*2:3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER: 3+NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4]    
    size_residual_normalized = K.reshape(size_residual_normalize, (-1, NUM_SIZE_CLUSTER, 3))
    size_residuals = size_residuals_normalized * K.expand_dims(K.constant(mean_size_arr, dtype=tf.float32), 0)
    return [center,heading_scores,heading_residual_normlized,heading_residual,size_scores,size_sresidual_normalized,size_resiudal]
   
     
    
    
'''   
    # ---- use slice function to separte the result ----
    #center calculate  
    center = Lambda(center_calculate,arguments={"inputs2":stage1_center,"begin":[0,0],"size": [-1,3]},name="get_center")(output)
    
    #heading calculate
    heading_scores = Lambda(slice_keras,arguments={"begin":[0,3],"size": [-1,NUM_HEADING_BIN]},name="get_heading_scores")(output)
    heading_residuals_normalized = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN],"size": [-1,NUM_HEADING_BIN]},name="heading_residuals_normalized")(output)
   
    heading_load_scores = Lambda(loding_heading_scores)(heading_scores)
    heading_load_residuals_normalized = Lambda(loding_heading_load_residuals_normalized)(heading_residuals_normalized)
    heading_residuals = Lambda(loding_heading_load_residuals,name="get_heading_residual")(heading_residuals_normalized)
    
    #size calculate 
    size_scores = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN*2],"size": [-1,NUM_SIZE_CLUSTER]})(output)
    size_residuals_normalized = Lambda(slice_keras,arguments={"begin":[0,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER],"size": [-1,NUM_SIZE_CLUSTER*3]})(output)
    size_residuals_normalized = Lambda(reshape_keras,arguments={"shape":[batchsize, NUM_SIZE_CLUSTER, 3]})(size_residuals_normalized)
    size_scores = Lambda(loding_size_scores)(size_scores)
    size_residuals_normalized = Lambda(loding_size_residuals_normalized)(size_residuals_normalized)
    size_residuals = Lambda(loding_size_residuals,arguments={"mean_size_arr" : mean_size_arr})(size_residuals_normalized)
'''
    args_loss = [center_label,
                 heading_class_label,
                 heading_residual_label,
                 size_class_label,
                 size_residual_label,
                 sem_cls_label,
                 box_label_mask,
                 vote_label,
                 vote_label_mask,
                 center,
                 heading_scores, 
                 heading_residuals_normalized,
                 size_score,
                 size_residual_normalized,
                 sem_cls_score,
                 seeds_xyz,
                 seeds_idx,
                 votes_xyz,
                 objectness_score,
                 proposals_xyz] # pack all arguments as a list
    # use Lambda layer to calculate the loss
    loss = layers.Lambda(f_pointnet_loss, output_shape=(10,),
                        arguments={'config':config},
                        name='f_pointnet_loss')(args_loss)
    
def create_Frustum_pointnet(k,num_class=10,num_points=2048):
    input_points = Input(shape = (num_points,6),dtype="float32")
    k =Input(shape=(k,))
    #inp = Reshape((num_points, 6, 1))(input_points)
    x,_ = instance_seg_pointnet(input_points, k)
    mask,point_cloud_xyz,point_cloud_xyz_stage1,mask_xyz_mean = masking_model(input_points,x)
    stage1_center=T_Net(point_cloud_xyz_stage1,mask_xyz_mean,k)
    output=Amodal_3dbox_estimation_pointnet(point_cloud_xyz,stage1_center,k)
    return Model(inputs=input_points,outputs = output  )#Model 可以input 多个吗？

pointnet=create_Frustum_pointnet(10)
#result = pointnet.predict(np.random.random((32,2048,6)))
pointnet.summary()
plot_model(pointnet, to_file='model.png')  