# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:22:44 2020

@author: chuhaofan
"""
import tensorflow as tf
import numpy as np
from loss_utils import huber_loss,get_box3d_corners_helper,
from loss_utils import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, compute_box3d_iou,mean_size_arr

def f_pointnet_loss(args, NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    '''
    for calculate the loss, following are needed. Be careful with the order!
    Args:
    Labels:
        0 - mask_label,
        1 - center_label, 
        2 - headheading_class_label,
        3 - heading_residual_label, 
        4 - size_class_label,
        5 - size_residual_label,
    Predictions:
        6 - logits
        7 - stage1_center,
        8 - center, 
        9 - heading_scores,
        10 - heading_residuals_normalized,
        11 - heading_residuals, 
        12 - size_scores,
        13 - size_residuals_normalized,
        14 - size_residuals
    
    Returns:
    0 - loss
    1 - 3d mask loss
    2 - center loss
    3 - stage1 center loss
    4 - heading class loss
    5 - heading residual normalized loss
    6 - size class loss
    7 - size residual normalized loss
    8 - corners loss
    '''
    # first unpack inputs
    # labels
    mask_label = args[0]
    center_label = args[1]
    heading_class_label = args[2]
    heading_residual_label = args[3]
    size_class_label = args[4]
    size_residual_label = args[5]

    # predictions and sample point
    logits = args[6]
    stage1_center = args[7]
    center = args[8]
    heading_scores = args[9]
    heading_residuals =args[10]
    heading_residuals_normalized = args[11]
    size_scores = args[12]
    size_residuals_normalized = args[13]
    size_residuals =args[14]
    
    # calculate the 3d mask loss 
    #batch_size = logits.get_shape()[0].value
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=mask_label))

    # calculate the box loss
    box_losses = compute_box_loss(mask_label,logits,
                                  center_label, center,stage1_center,
                                  heading_class_label,heading_scores,
                                  heading_residual_label,heading_residuals_normalized, 
                                  size_class_label,size_scores,
                                  size_residual_label,size_residuals_normalized,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr)#config 作用
       
    # unpack the loss
    center_loss,stage1_center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss = box_losses
    # calculate define box_loss_sum
    box_loss = center_loss + heading_class_loss + size_class_loss + heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + stage1_center_loss
    
    #calculate the corner loss
    corners_loss = compute_corners_loss()
    
    
    loss = mask_loss +  box_loss*0.1 + corners_loss
    
    # mask_loss + (center_loss + heading_class_loss + size_class_loss + heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + stage1_center_loss)*0.1 + corners_loss

    return loss

def compute_box_loss(mask_label,logits,
                     center_label, center,stage1_center,
                     heading_class_label,heading_scores,heading_residual_label,heading_residuals_normalized,
                     size_class_label,size_scores,size_residual_label,size_residuals_normalized,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    center_dist = tf.norm(center_label - center, axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)

    stage1_center_dist = tf.norm(center_label - stage1_center, axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

    heading_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=heading_scores, labels=heading_class_label))

    tmp = tf.one_hot(heading_class_label, depth=NUM_HEADING_BIN, on_value=1, off_value=0, axis=-1) # BxNUM_HEADING_BIN
    #print(tmp)
    heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(heading_residuals_normalized*tf.to_float(tmp), axis=1) - heading_residual_normalized_label, delta=1.0)
    

    size_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=size_scores, labels=size_class_label))

    tmp2 = tf.one_hot(size_class_label, depth=NUM_SIZE_CLUSTER, on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
    tmp2_tiled = tf.tile(tf.expand_dims(tf.to_float(tmp2), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum(size_residuals_normalized*tmp2_tiled, axis=[1]) # Bx3

    tmp3 = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32),0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum(tmp2_tiled * tmp3, axis=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
 
    size_normalized_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    return [center_loss,stage1_center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss,tmp1,tmp2,tmp3 ]
    
def compute_corners_loss(center,heading_residuals,size_residuals,center_label, heading_label, size_label,tmp,tmp2,tmp3,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,mean_size_arr):
    corners_3d = get_box3d_corners(center, heading_residuals, size_residuals) # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(tmp, 2), [1,1,NUM_SIZE_CLUSTER]) * tf.tile(tf.expand_dims(tmp2,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum(tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask,-1),-1))*corners_3d, axis=[1,2]) # (B,8,3)

    heading_bin_centers = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    heading_label = tf.expand_dims(heading_residual_label,1) + tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(tmp)*heading_label, 1)
    mean_sizes = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0) # (1,NS,3)
    size_label = mean_sizes + tf.expand_dims(size_residual_label, 1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.to_float(tmp2),-1)*size_label, axis=[1]) # (B,3)
    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1), tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
    return corners_loss
    