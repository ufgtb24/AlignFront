import tensorflow as tf
from tensorflow.contrib import slim

from my_batch_norm import bn_layer_top


class discriminator:
    def __init__(self):
        pass
    
    def extract_layer(self,input,new_u,new_r):
        '''
        :param input: [batch_size,r,u]
        :param new_u: dim of encoding
        :param new_r: main structure number
        :return:
        '''
        net_u=slim.fully_connected(input,new_u)
        #[batch_size,r,new_u]
        net_u = slim.dropout(net_u, keep_prob=0.5, scope='Dropout_u')
        
        #[batch_size,u,r]
        net_r=tf.transpose(input,[0,2,1])
        net_r=slim.fully_connected(net_r,new_r)
        #[batch_size,u,new_r]
        net_r = slim.dropout(net_r, keep_prob=0.5, scope='Dropout_r')
        
        return net_u,net_r
            
        
    def extract(self,seq,da,r):
        '''
        
        :param seq: [batch_size,n,u]
        :param da:
        :param r:
        :return:
        '''
        with tf.variable_scope('extractor'):
            with slim.arg_scope([slim.fully_connected],normalizer_fn=bn_layer_top):
                #[batch_size,n,da] rank 1 共享变量
                net=slim.fully_connected(seq,da)
                net=slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
                #[batch_size,n,r]
                net=slim.fully_connected(net,r,activation_fn=None)
                #[batch_size,n,r] n=6
                sum_weight=tf.nn.softmax(net,axis=-2)
                #[batch_size,r,u]
                M=tf.matmul(sum_weight,seq,transpose_a=True)
                
                net_u,net_r=self.extract_layer(M,32,8)
                net_uu,net_ur=self.extract_layer(net_u,32,8)
                net_ru,net_rr=self.extract_layer(net_r,32,8)
                net=tf.concat([slim.flatten(net_uu),
                           slim.flatten(net_ur),
                           slim.flatten(net_ru),
                           slim.flatten(net_rr)],
                          axis=-1)
                net=slim.fully_connected(net,1,activation_fn=None,normalizer_fn=None)
                output=tf.nn.sigmoid(net)
                return output

