import tensorflow as tf
from tensorflow.contrib import layers,slim

from my_batch_norm import bn_layer_top


class discriminator:
    def __init__(self):
        pass
    
    def extract_layer(self,input,u,r):
        net_u=layers.fully_connected(input,u)
        net_u = layers.dropout(net_u, keep_prob=0.5, scope='Dropout_u')
        
        net_r=tf.transpose(input,[0,2,1])
        net_r=layers.fully_connected(net_r,r)
        net_r = layers.dropout(net_r, keep_prob=0.5, scope='Dropout_r')
        
        return net_u,net_r
            
        
    def extract(self,seq,da,r):
        '''
        
        :param seq: [batch_size,n,u]
        :param da:
        :param r:
        :return:
        '''
        with tf.variable_scope('extractor'):
            with slim.arg_scope([layers.fully_connected],normalizer_fn=bn_layer_top):
                #[batch_size,n,da]
                net=layers.fully_connected(seq,da)
                net=layers.dropout(net, keep_prob=0.5, scope='Dropout_1b')
                #[batch_size,n,r]
                net=layers.fully_connected(net,r,activation_fn=None)
                #[batch_size,n,r] n=6
                sum_weight=tf.nn.softmax(net,axis=-2)
                #[batch_size,r,u]
                M=tf.matmul(sum_weight,seq,transpose_a=True)
                
                net_u,net_r=self.extract_layer(M,32,8)
                net_uu,net_ur=self.extract_layer(net_u,32,8)
                net_ru,net_rr=self.extract_layer(net_r,32,8)
                net=tf.concat([layers.flatten(net_uu),
                           layers.flatten(net_ur),
                           layers.flatten(net_ru),
                           layers.flatten(net_rr)],
                          axis=-1)
                net=layers.fully_connected(net,1,activation_fn=None,normalizer_fn=None)
                output=tf.nn.sigmoid(net)
                return output
