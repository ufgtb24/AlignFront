import tensorflow as tf
from histogram import Histogrammer as Hist
import numpy as np
class RelationNet(object):
    '''
    需要建立四种 g_function，UL,UR,DL,DR，暂时不需要更细分
    '''
    def __init__(self,trans_box,mask,id2idx_dict):
        '''
        
        :param trans_box:
        :param mask: [batch,seqlen(28)]
        :param id2idx_dict: a list translate teeth_id, to trans_box index
        '''
        self.hist=Hist
        self.trans_box=trans_box
        self.mask=mask
        
        
    def get_idx(self,pair):
        def get_idx(id):
            if id<16:
                return id-2
            else:
                return id-4
        return get_idx(pair[0]),get_idx(pair[1])
    
    def build_histogram(self):
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
        r = r1 + r2 + r3 + r4 + r5
        
        
        # exist_pairs= [i for i in r if self.judge_exist(i)]
        # tf.map_fn()
        histogram_list=[]
        for pair in r:
            # 针对已经变换过的牙齿对，进行判断
            histogram_list.append(self.hist.histogram(self.get_idx(pair)))
        return histogram_list









































