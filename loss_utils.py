# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:23:03 2020

@author: chuhaofan
"""
import tensorflow as tf
import numpy as np
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

mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))#(NS,3)
for i in range(NUM_SIZE_CLUSTER):
    mean_size_arr[i,:] = type_mean_size[class2type[i]]#按物体类别排序的anchorbox 大小
'''
[[2.114256 1.6203   0.927272]
 [0.791118 1.279516 0.718182]
 [0.923508 1.867419 0.845495]
 [0.591958 0.552978 0.827272]
 [0.699104 0.454178 0.75625 ]
 [0.69519  1.346299 0.736364]
 [0.528526 1.002642 1.172878]
 [0.500618 0.632163 0.683424]
 [0.404671 1.071108 1.688889]
 [0.76584  1.398258 0.472728]]
'''
def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)
"""
def huber_loss(error, delta=1.0):
    abs_error = K.abs(error)
    quadratic = K.clip(abs_error, max_value=delta, min_value=-1)#逐元素clip（将超出指定范围的数强制变为边界值）
    linear = (abs_error -quadratic)
    loss = 0.5*K.square(quadratic) + delta * linear
    return loss
"""
def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    print('-----', centers)
    N = centers.get_shape()[0].value
    l = tf.slice(sizes, [0,0], [-1,1]) # (N,1) begin [0,0]即N=0 0  size [-1,1]
    w = tf.slice(sizes, [0,1], [-1,1]) # (N,1)
    h = tf.slice(sizes, [0,2], [-1,1]) # (N,1)
    print(l,w,h)
    x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)  8个(N,1) concatenate
    y_corners = tf.concat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1) # (N,8)  8个（N，1） concatenate
    z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)  8个（N，1） concatenate
    corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    print(x_corners, y_corners, z_corners)
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c,zeros,s], axis=1) # (N,3)#数据列写入 m每列数据的元素进行行连接
    row2 = tf.stack([zeros,ones,zeros], axis=1)
    row3 = tf.stack([-s,zeros,c], axis=1)
    R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)#通过增加的维度 再对指定的维度进行叠加
    print(row1, row2, row3, R, N)
    corners_3d = tf.matmul(R, corners) # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3) 
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0].value
    heading_bin_centers = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,) NH:BH NUM_HEADING_BIN
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0) 
    # (B,NH) 增加的维度大小与heading_residuals: (B,NH)的大小自适应
    
    mean_sizes = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0) + size_residuals # (B,NS,3)
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes,1), [1,NUM_HEADING_BIN,1,1]) # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings,-1), [1,1,NUM_SIZE_CLUSTER]) # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center,1),1), [1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1]) # (B,NH,NS,3)

    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N,3]), tf.reshape(headings, [N]), tf.reshape(sizes, [N,3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])

def get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, logits,\
             end_points, reg_weight=0.001):
    """ logits: BxNxC,
        mask_label: BxN, """
    batch_size = logits.get_shape()[0].value
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)

    stage1_center_dist = tf.norm(center_label - end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    heading_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    tmp = tf.one_hot(heading_class_label, depth=NUM_HEADING_BIN, on_value=1, off_value=0, axis=-1) # BxNUM_HEADING_BIN
    print(tmp)
    heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(end_points['heading_residuals_normalized']*tf.to_float(tmp), axis=1) - heading_residual_normalized_label, delta=1.0)
    print(heading_residual_normalized_loss)
    tf.summary.scalar('heading residual normalized loss', heading_residual_normalized_loss)

    size_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    tmp2 = tf.one_hot(size_class_label, depth=NUM_SIZE_CLUSTER, on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
    tmp2_tiled = tf.tile(tf.expand_dims(tf.to_float(tmp2), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum(end_points['size_residuals_normalized']*tmp2_tiled, axis=[1]) # Bx3

    tmp3 = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32),0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum(tmp2_tiled * tmp3, axis=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
 
    size_normalized_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    print(size_residual_normalized_loss)
    tf.summary.scalar('size residual normalized loss', size_residual_normalized_loss)

    # Compute IOU 3D
    iou2ds, iou3ds = tf.py_func(compute_box3d_iou, [end_points['center'], end_points['heading_scores'], end_points['heading_residuals'], end_points['size_scores'], end_points['size_residuals'], center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label], [tf.float32, tf.float32])
    tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
    tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))
 
    end_points['iou2ds'] = iou2ds 
    end_points['iou3ds'] = iou3ds 

    # Compute BOX3D corners
    corners_3d = get_box3d_corners(end_points['center'], end_points['heading_residuals'], end_points['size_residuals']) # (B,NH,NS,8,3)
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
    print("Corners dist: ", corners_dist)
    corners_loss = huber_loss(corners_dist, delta=1.0) 
    tf.summary.scalar('corners loss', corners_loss)

    return mask_loss + (center_loss + heading_class_loss + size_class_loss + heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + stage1_center_loss)*0.1 + corners_loss
