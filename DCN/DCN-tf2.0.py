# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
import pickle
from util.train_model import train_test_model_demo


class CrossLayer(tf.keras.layers.Layer):
    def __init__(self,output_dim,num_layer,**kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.input_dim = input_shape[2]
        # print(self.input_dim)
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape=[1,self.input_dim],initializer = 'glorot_uniform',name='w_{}'.format(i),trainable=True))
            self.bias.append(self.add_weight(shape=[1,self.input_dim],initializer = 'zeros',name='b_{}'.format(i),trainable=True))
        self.built = True

    def call(self,input):
        # 按照论文的公式
        # x0 = tf.einsum('bij->bji',input) # output[j][i] = m[i][j]
        # print("x0_shape",x0.get_shape())# (9, 390, 1)
        # x1 = tf.einsum('bmn,bkm->bnk', input, x0)
        # print("x1_shape", x1.get_shape()) # (9, 390, 390)
        # print("self.W[0]_shape", self.W[0].get_shape())
        # cross = tf.einsum('bmn,kn->bkm',x1,self.W[0]) + self.bias[0] + input
        # print("cross0", cross.get_shape())# (9, 1, 390)
        # for i in range(1,self.num_layer):
        #     x0 = tf.einsum('bij->bji',cross) # output[j][i] = m[i][j]
        #     x1 = tf.einsum('bmn,bkm->bnk',input,x0)
        #     cross = tf.einsum('bmn,kn->bkm',x1,self.W[i]) + self.bias[i] + cross

        # 优化论文公式 改变结合律
        x0 = tf.einsum('bij->bji',input) # output[j][i] = m[i][j]
        x1 = tf.einsum('bmn,km->bnk', x0, self.W[0])
        cross = tf.einsum('bkm,bnk->bnm',input,x1) + self.bias[0] + input
        for i in range(1,self.num_layer):
            x0 = tf.einsum('bij->bji',cross) # output[j][i] = m[i][j]
            x1 = tf.einsum('bmn,km->bnk', x0, self.W[i])
            cross = tf.einsum('bkm,bnk->bnm', cross,x1) + self.bias[i] + cross
        return cross
        
class Deep(tf.keras.layers.Layer):
    def __init__(self,dropout_deep,deep_layer_sizes):
        # input_dim = num_size + embed_size = input_size
        super(Deep, self).__init__()
        self.dropout_deep = dropout_deep
        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i),tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i),tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i),tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i),tf.keras.layers.Dropout(dropout_deep[i]))
        # last layer
        self.fc = tf.keras.layers.Dense(128,activation=None,use_bias=True)
        
    def call(self,input):
        y_deep = getattr(self,'dense_' + str(0))(input)
        y_deep = getattr(self,'batchNorm_' + str(0))(y_deep)
        y_deep = getattr(self,'activation_' + str(0))(y_deep)
        y_deep = getattr(self,'dropout_' + str(0))(y_deep)
        
        for i in range(1,len(self.deep_layer_sizes)):
            y_deep = getattr(self,'dense_' + str(i))(y_deep)
            y_deep = getattr(self,'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self,'activation_' + str(i))(y_deep)
            y_deep = getattr(self,'dropout_' + str(i))(y_deep)
        
        output = self.fc(y_deep)
        return output
    
class DCN(tf.keras.Model):
    def __init__(self,num_feat,num_field,dropout_deep,deep_layer_sizes,embedding_size=10):
        super().__init__()
        self.num_feat = num_feat # F =features nums
        self.num_field = num_field # N =fields of a feature 
        self.dropout_deep  = dropout_deep
        
        # Embedding 这里采用embeddings层因此大小为F* M F为特征数量，M为embedding的维度
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform') # F * M 
        self.feat_embeddings = feat_embeddings
        
        self.crosslayer = CrossLayer(output_dim = 128,num_layer=8)
    
        self.deep = Deep(dropout_deep,deep_layer_sizes)
        self.fc = tf.keras.layers.Dense(1,activation='sigmoid',use_bias=True)
        
    def call(self,feat_index,feat_value):
        
        # embedding part  feat_index = inputs为输入 feat_embeddings为一个layer。
        feat_embedding_0 = self.feat_embeddings(feat_index) # Batch * N * M 
#         print(feat_value.get_shape())
        feat_embedding = tf.einsum('bnm,bn->bnm',feat_embedding_0,feat_value)
        # print("feat_embedding:",feat_embedding.get_shape()) # 32 * 39 * 10
        stack_input = tf.keras.layers.Reshape((1,-1))(feat_embedding)
        # print("stack_input:",stack_input.get_shape()) # 32 * 1 * 390
        
        x1 = self.crosslayer(stack_input)
        x2 = self.deep(stack_input)
        
        x3 = tf.keras.layers.concatenate([x1,x2],axis=-1)
        output = self.fc(x3)
        return output

if __name__ == '__main__':
    AID_DATA_DIR = "../data/Criteo/"
    feat_dict_ = pickle.load(open(AID_DATA_DIR + '/feat_dict_10.pkl2', 'rb'))

    dcn = DCN(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400])

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    train_test_model_demo(dcn,train_label_path, train_idx_path, train_value_path)