import tensorflow as tf
import numpy as np
def kernel_mat(index):
    '''
    :param index: [num_batch,line_num,step_num,3] 需要被求kernel的点坐标
    :return: [num_batch,line_num,step_num,27,3]
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
    index=tf.expand_dims(index, axis=-2) #[num_batch,line_num,step_num,1,3]
    index=index+kernel #[num_batch,line_num,step_num,27,3]
    

    return index


class Histogrammer:
    def __init__(self, box, center_pos, tooth_ids):
        #[batch_num,28,w,h,d]
        # if a tooth is absence, its value is zero padded, tooth_num is fixed of 24
        self.box=box
        self.center_pos=center_pos
        # [60 ] 0.2mm
        self.center_list=np.linspace(-19,39,2)
        
        
    def distance_mulpt(self, tooth_pair):
        '''
        get the vector of distance between specified two teeth
        :param tooth_pair:
        :return:[num_batch,plain_pt_num]
        '''
        #[batch,w,h,d]
        box1=self.box[:,tooth_pair[0]]
        box2=self.box[:,tooth_pair[1]]
        #[batch,3]
        pos1= self.center_pos[:, tooth_pair[0]]
        pos2= self.center_pos[:, tooth_pair[1]]
        # [num_batch,plain_pt_num]
        pts_2teeth=self.dis_between_box(box1, pos1, box2, pos2)
        return pts_2teeth
    
    def histogram(self, tooth_pair):
        '''
        get histogram of distance between two teeth count on given range
        :param tooth_pair:
        :param center_list: a list [num_centers]
        :return: [num_batch,num_centers]
        '''
        internal= self.center_list[1] - self.center_list[0]
        # [num_batch,plain_pt_num,num_centers]
        d=tf.expand_dims(self.distance_mulpt(tooth_pair), axis=-1)
        # [num_centers]
        c=tf.constant(self.center_list)
        # [num_batch,num_centers]
        h=tf.reduce_sum(tf.minimum(tf.abs(d-c)/internal,0),axis=1)
        return h
    
    def dis_between_box(self, box1, pos1, box2, pos2):
        '''
         get distance between corresponding points on two plains vertical to the direction from
         one tooth to another

        :param box1: [batch,w,h,d]
        :param pos1: [batch,3]
        :param box2: [batch,w,h,d]
        :param pos2: [batch,3]
        :return dis_pts: [num_batch,plain_pt_num]
        '''
        distance=tf.sqrt(tf.reduce_sum(tf.squared_difference(pos1,pos2),axis=1,keepdims=True))  #[batch,1]
        # direction =tf.norm(pos2-pos1,axis=1)
        direction=(pos2-pos1)/distance
        batch_num=tf.shape(box1)[0]
        z_vec_0=tf.zeros([batch_num,2])
        z_vec_1=tf.ones([batch_num,1])
        z_vec=tf.concat([z_vec_0,z_vec_1],axis=1) # [batch,3]
        plain_vec1=tf.norm(tf.cross(direction, z_vec), axis=1) # [batch,3]
        plain_vec2=tf.norm(tf.cross(direction, plain_vec1), axis=1) # [batch,3]
        box1_c2e=self.dis_c2e(box1,direction,plain_vec1,plain_vec2) #[num_batch,plain_pt_num]
        box2_c2e=self.dis_c2e(box2,-direction,plain_vec1,plain_vec2) #[num_batch,plain_pt_num]
        dis_pts=tf.expand_dims(distance,axis=-1)-box1_c2e-box2_c2e
        return dis_pts

    def sample(self,box, indices):
        '''
        calculate the kernel value of a sample line
        :param box: [batch_num,w,h,d]
        :param indices: [num_batch,line_num,step_num,3]
        :return: [num_batch,line_num,step_num]
        '''
        num_batch, line_num, step_num, _ = tf.shape(indices)
        batch_idx = tf.range(0, num_batch)
        batch_idx = tf.reshape(batch_idx, (num_batch, 1, 1, 1, 1))
    
        # [num_batch,line_num,step_num,27,3]
        kernel_indices = kernel_mat(indices)
    
        b = tf.tile(batch_idx, (1, line_num, step_num, 27, 1))
        # [num_batch,line_num,step_num,27,4]
        kernel_indices = tf.concat([b, kernel_indices], -1)
        # [num_batch,line_num,step_num,27]
        kernel_sample = tf.gather_nd(box, kernel_indices)
        # [num_batch,line_num,step_num]
        kernel_value = tf.reduce_sum(kernel_sample, axis=-1)
        # [num_batch,line_num,step_num]
        center_sample = tf.gather_nd(box, indices)
        # sum(X(i)+X(i+k),k)   [num_batch,line_num,step_num]
        value = 27 * center_sample - kernel_value
        return value

    def dis_c2e(self, box, step_direction, plain_vec1, plain_vec2):
        '''
        distance from center to edge
        :param step_direction: [batch,3]
        :param plain_vec1: [batch,3]
        :param plain_vec2: [batch,3]
        :return:edge_dis [num_batch,plain_pt_num]
        '''
        batch_num=tf.shape(box)[0]
        x=tf.linspace(-3,3,1)
        y=tf.linspace(-3,3,1)
        # [7,7]
        x_t, y_t = tf.meshgrid(x, y)
        x_t_flat = tf.reshape(x_t, [-1])  # [plain_pt_num]
        y_t_flat = tf.reshape(y_t, [-1])
        xy_flat=tf.expand_dims(tf.stack([x_t_flat,y_t_flat],axis=-1)) #[1,plain_pt_num,2]
        xy_flat=tf.tile(xy_flat,tf.stack([batch_num, 1, 1])) #[batch,plain_pt_num,2]
        vec_flat=tf.stack([plain_vec1,plain_vec2],axis=1) #[batch,2,3]
        center=tf.constant([64,64,64],dtype=tf.float32)
        pts_on_plain=tf.matmul(xy_flat,vec_flat)+center #[batch,plain_pt_num,3]
        pts_on_plain=tf.expand_dims(pts_on_plain,axis=2) #[batch,plain_pt_num, 1, 3]
        # the range to detect edge distance
        step=tf.range(10,64) #[step_num]
        step_direction=tf.expand_dims(step_direction, axis=1)  #[batch, 1, 3]
        # [batch,plain_pt_num, step_num, 3]
        pts_on_line=pts_on_plain+tf.expand_dims(step_direction * tf.expand_dims(step,axis=-1), axis=1)
        pts_on_line=tf.to_int32(pts_on_line)
        kernel_value=self.sample(box,pts_on_line) #[num_batch,plain_pt_num,step_num]
        kernel_mul_dis=kernel_value*step #[num_batch,plain_pt_num,step_num]
        edge_dis=tf.reduce_sum(tf.minimum(kernel_mul_dis,1),axis=-1) #[num_batch,plain_pt_num]
        return edge_dis
        
        
        
        


        
        
        
        
