import tensorflow as tf
from histogram import Histogrammer as Hist
class RelationNet(object):
    '''
    需要建立四种 g_function，UL,UR,DL,DR，暂时不需要更细分
    '''
    def __init__(self,box):
        '''
        
        :param box_up: [batch_size,up_size,w,h,d]
        :param box_low:  [batch_size,low_size,w,h,d]
        '''
        self.hist=Hist
        self.box=box
        self.phase=phase
        self.upper_feature=upper_feature#[b,num/2,fc_size+4]
        self.lowwer_feature=lowwer_feature
        
    def get_box(self,id):
        if id<=15:
            return self.box[:,]
        
    def build_pairs(self):
        '''
        2  - 15, 31 - 18
        :return: a list of tuple
        '''
        up_id = list(range(2, 16))
        low_id = list(range(31, 17, -1))
        r1 = list(zip(up_id, low_id))
        r2 = list(zip(up_id[1:7], low_id[:6]))
        r3 = list(zip(up_id[-7:-1], low_id[-6:]))
        r4 = list(zip(up_id[1:], up_id[:-1]))
        r5 = list(zip(low_id[1:], low_id[:-1]))
        self.r = r1 + r2 + r3 + r4 + r5
    
    def histogram_loss(self):
        self.
        
        

    def triangle_net(self, long, short, scope):
        '''
        :param long: [b, half_arch_size, obj_size]
        :param short: [b, half_arch_size-1, obj_size]
        :return:
        '''
        with tf.variable_scope(scope):
            with tf.variable_scope('build_triangle'):
                # part.shape = [batch_size, triangle_num,3*obj_size]
                triangles=tf.concat([long[:,:-1,:],short,long[:,1:,:]],axis=2)

                # triangles    [batch_size* triangle_num, 3*obj_size]
                triangle_flat=tf.reshape(triangles,shape=[-1,triangles.shape[2].value])

            # triangles    [batch_size* triangle_num, g_size]
            triangle_flat=self.g_net(triangle_flat)
            return triangle_flat

    def g_net(self, triangle_flat):
        with tf.variable_scope('g_net'):
            units = self.params.g_units
            fc = commen.dense_relu_batch(triangle_flat, output_size=units, phase=self.phase, scope='fc_1')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_2')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_3')
            return fc #[b,g_size]

    def build_relation(self):
        '''
        输入方向必须是
        3  - 14
        30 - 19
        :param box: conv fc 之后的单个物体集合 [b,24,conv_fc]
        :param pos: 物体的box下角位置集合
        :return: [Tensor(b,g_output_size)] x11
        '''
        triangle_num=int(self.upper_feature.shape[1].value/2)
        part_size=triangle_num+1

        with tf.variable_scope('devide_parts'):
            UL=self.upper_feature[:,:part_size,:]
            UR=self.upper_feature[:,-part_size:,:]
            DL=self.lowwer_feature[:,:part_size,:]
            DR=self.lowwer_feature[:,-part_size:,:]

        with tf.variable_scope('triangle_feature'):
            # g_part    [batch_size * triangle_num, g_output_size]
            triangles_1 =self.triangle_net(UL, DL[:, :-1, :], scope='triangle_1') # 3-30-4 to 8-25-9
            triangles_2 =self.triangle_net(DL, UL[:, 1:, :], scope='triangle_2') # 30-4-29 to 25-9-24
            triangles_3 =self.triangle_net(UR, DR[:, 1:, :], scope='triangle_3') # 8-24-9 to 13-19-14
            triangles_4 =self.triangle_net(DR, UR[:, :-1, :], scope='triangle_4')# 25-8-24 to 20-13-19

        with tf.variable_scope('extract_from_triangle'):
            #left,mid,right are all of shape [batch_size, triangle_num,extract_size]
            left_1,mid_1,right_1=self.extract_3_tooth(triangles_1,triangle_num,scope='triangle_1')
            left_2,mid_2,right_2=self.extract_3_tooth(triangles_2,triangle_num,scope='triangle_2')
            left_3,mid_3,right_3=self.extract_3_tooth(triangles_3,triangle_num,scope='triangle_3')
            left_4,mid_4,right_4=self.extract_3_tooth(triangles_4,triangle_num,scope='triangle_4')

        with tf.variable_scope('combineed_feature_to_tooth'):
            with tf.variable_scope('part_ul'):
                # shape = [batch_size, triangle_num-1,3*extract_size]
                # 没有考虑后牙，所以是triangle_num -1
                # 将从三组关系中提取出的关于某颗牙的特征组合起来
                UL_relation_feature=tf.concat([right_1[:,:-1],mid_2[:,:-1],left_1[:,1:]],axis=2)
                # ul_combined shape=[batch_size, tooth_num/4-1,combined_units]
                ul_combined=self.combine_feature(UL_relation_feature,tooth_num=triangle_num -1)

            with tf.variable_scope('part_dl'):
                DL_relation_feature=tf.concat([right_2[:,:-1],mid_1[:,1:],left_2[:,1:]],axis=2)
                dl_combined=self.combine_feature(DL_relation_feature,tooth_num=triangle_num -1)

            with tf.variable_scope('part_ur'):
                UR_relation_feature=tf.concat([right_3[:,:-1],mid_4[:,1:],left_3[:,1:]],axis=2)
                ur_combined=self.combine_feature(UR_relation_feature,tooth_num=triangle_num -1)

            with tf.variable_scope('part_dr'):
                DR_relation_feature=tf.concat([right_4[:,:-1],mid_3[:,:-1],left_4[:,1:]],axis=2)
                dr_combined=self.combine_feature(DR_relation_feature,tooth_num=triangle_num -1)

        with tf.variable_scope('rns_featrue_for_arch'):
            # 除了2颗后牙之外的上牙序列 [batch_size, tooth_num/2-2,combined_units]
            # 可以用来和上牙 lstm 组合
            relation_upper=tf.concat([ul_combined,ur_combined],axis=1)
            relation_lowwer=tf.concat([dl_combined,dr_combined],axis=1)
            return [relation_upper,relation_lowwer]


    def combine_feature(self,relation_feature,tooth_num):
        '''
        将组合的关于单牙的feature 经过fc
        :param relation_feature:
        :param tooth_num:
        :param scope:
        :return:
        '''
        units=relation_feature.shape[2].value
        relation_feature=tf.reshape(relation_feature,shape=[-1,units])
        with tf.variable_scope('combine_net'):
            units = self.params.combined_units
            fc = commen.dense_relu_batch(relation_feature, output_size=units, phase=self.phase, scope='fc_1')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_2')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_3')
            # [batch_size , triangle_num, extract_size]
            fc=tf.reshape(fc,shape=[-1,tooth_num,units])
            return fc



    def extract_fc(self,g_outputs,triangle_num,scope):
        with tf.variable_scope(scope):
            units = self.params.extract_units
            fc = commen.dense_relu_batch(g_outputs, output_size=units, phase=self.phase, scope='fc_1')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_2')
            fc = commen.dense_relu_batch(fc, output_size=units, phase=self.phase, scope='fc_3')
            # [batch_size , triangle_num, extract_size]
            fc=tf.reshape(fc,shape=[-1,triangle_num,units])
            return fc

    def extract_3_tooth(self,g_outputs,triangle_num,scope):
        '''
        从concat的关系组分别提取单个牙，并返回三个牙的fc的list
        :param g_outputs: [batch_size * triangle_num , g_output_size]
        :param scope: 上一层的范围
        :return: a list for 3 indivifual teeth fc
                of shape [batch_size , triangle_num , extracted_size]
        '''
        tooth_fc=[]
        with tf.variable_scope(scope):
            tooth_fc.append(self.extract_fc(g_outputs,triangle_num,scope='for_tooth1'))
            tooth_fc.append(self.extract_fc(g_outputs,triangle_num,scope='for_tooth2'))
            tooth_fc.append(self.extract_fc(g_outputs,triangle_num,scope='for_tooth3'))
            return  tooth_fc







































