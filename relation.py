import tensorflow as tf
import numpy as np
def kernel_mat(index):
    '''
    :param index: [num_batch,line_num,line_pt_num,3] 需要被求kernel的点坐标
    :return: [num_batch,line_num,line_pt_num,27,3]
    box中index处需要采样的所有kernel点的坐标，周围9个点（包括自己）
    '''
    x = [-1,0,1]
    y = [-1,0,1]
    z = [-1,0,1]
    np.meshgrid()
    x_t, y_t ,z_t= tf.meshgrid(x, y,z)
    # flatten
    x_t_flat = tf.reshape(x_t, [-1])  # [27]
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])
    
    kernel = tf.stack([x_t_flat, y_t_flat, z_t_flat],axis=-1)  # [27,3]
    index=tf.expand_dims(index, axis=-2) #[num_batch,line_num,line_pt_num,1,3]
    index=index+kernel #[num_batch,line_num,line_pt_num,27,3]
    

    return index

def sample(box,indices):
    '''
    
    :param box: [batch_num,l,l,l]
    :param indices: [num_batch,line_num,line_pt_num,3]
    :return: [num_batch,line_num,point_num]
    '''
    num_batch, line_num, line_pt_num, _ = tf.shape(indices)
    batch_idx = tf.range(0, num_batch)
    batch_idx = tf.reshape(batch_idx, (num_batch, 1, 1,1,1))
    
    # [num_batch,line_num,line_pt_num,27,3]
    kernel_indices=kernel_mat(indices)
    
    b = tf.tile(batch_idx, (1, line_num, line_pt_num, 27,1))
    # [num_batch,line_num,line_pt_num,27,4]
    kernel_indices = tf.concat([b, kernel_indices], -1)
    # [num_batch,line_num,line_pt_num,27]
    kernel_sample=tf.gather_nd(box,kernel_indices)
    # [num_batch,line_num,line_pt_num]
    kernel_value=tf.reduce_sum(kernel_sample,axis=-1)
    # [num_batch,line_num,line_pt_num]
    center_sample=tf.gather_nd(box,indices)
    # sum(X(i)+X(i+k),k)   [num_batch,line_num,line_pt_num]
    value=27*center_sample-kernel_value
    
    return value

class relation_net:
    def __init__(self,box,tooth_ids):
        #[batch_num,tooth_num,w,h,d]
        # if a tooth is absence, its value is
        self.box=box
        # if a tooth is absence, its id is 0
        self.ids=tooth_ids
        
    
def relation_loss(tooth_pair):
    t

def dis_c2e(box,direction,sample_length):
    '''
    distance from center to edge
    :param direction: [batch,3]
    :return:
    '''
    # indices[num_batch,line_num,line_pt_num,3]
    step=
    sample