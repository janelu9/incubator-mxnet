# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:11:46 2017

@author: T800
"""

import mxnet as mx
import numpy as np
  
def gen_buckets_data(data,batch_size=32):
    bucket_data={}
    for d in data:
        len_d=len(d[0]) 
        if len_d !=0:
            if  len_d in bucket_data:
                bucket_data[len_d].append(d)
            else:
                bucket_data[len_d]=[d]
    bk=bucket_data.keys()
    len_max=max(bk)
    for h,ij in enumerate(bucket_data.iteritems()):
        i,j=ij
        lj=len(j)
        rm_num=lj%batch_size
        if rm_num and i<len_max:
            bucket_data[bk[h+1]].extend(bucket_data[i][:rm_num])
            del bucket_data[i][:rm_num]
    for k in bucket_data.keys():
        if bucket_data[k]==[]: 
            bucket_data.pop(k)
    bucket_data=bucket_data.values()
    if rm_num>0:bucket_data[-1].extend(bucket_data[-2][:batch_size-rm_num])
    return bucket_data

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label,bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names     
        self.bucket_key=bucket_key
        self.pad=0

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]
    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
    
class BucketSentenceIter(mx.io.DataIter):
    def __init__(self,data,init_states,batch_size):
        super(BucketSentenceIter, self).__init__()
        data=gen_buckets_data(data,batch_size)
        self.data,self.label,self.bucket_plan,self.bucket_idx=[],[],[],[]
        for idx,i in enumerate(data):
            temp_data=np.zeros((len(i),len(i[0][0])),int)
            temp_label=np.zeros((len(i),),int)
            for jdx,j in enumerate(i):
                temp_data[jdx,:len(j[0])]=j[0]
                temp_label[jdx]=j[1]
            self.data.append(temp_data)
            self.label.append(temp_label)
            li=len(i)
            self.bucket_plan.extend([idx]*(li/batch_size))
            rg=range(0,li,batch_size)
            np.random.shuffle(rg)
            self.bucket_idx.append(rg)
        np.random.shuffle(self.bucket_plan)
        self.init_states=init_states
        self.batch_size=batch_size
        self.bucket_idx_ct=[0]*len(data)
        self.buckets=[len(i[0][0]) for i in data]
        self.default_bucket_key = self.buckets[-1]
        self.init_states_names=[i[0] for i in self.init_states]
        self.init_states_arrays=[mx.nd.zeros(x[1]) for x in self.init_states]
        self.provide_data = [('data', (self.batch_size, self.default_bucket_key))]+self.init_states
        self.provide_label = [('softmax_label', (self.batch_size,))]

    def __iter__(self):
        for i in self.bucket_plan:
            begin=self.bucket_idx[i][self.bucket_idx_ct[i]]
            data=self.data[i][begin:begin+self.batch_size]
            label=self.label[i][begin:begin+self.batch_size]
            self.bucket_idx_ct[i]+=1
            data=[mx.nd.array(data)]+self.init_states_arrays
            label = [mx.nd.array(label)]
            data_names = ['data']+self.init_states_names
            label_names = ['softmax_label']
            yield SimpleBatch(data_names, data, label_names, label,self.buckets[i])
    def reset(self):
        self.bucket_idx_ct = [0]*len(self.data)
        
        
        
    
        

