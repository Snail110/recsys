# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter

class CrossLayer(tf.keras.layers.Layer):
    def __init__(self,output_dim,num_layer,**kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.input_dim = input_shape[1]
        # print(self.input_dim)
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape=[self.input_dim,1],initializer = 'glorot_uniform',name='w_{}'.format(i),trainable=True))
            self.bias.append(self.add_weight(shape=[self.input_dim,1],initializer = 'zeros',name='b_{}'.format(i),trainable=True))
        self.built = True
    def call(self,input):

        x0 = tf.einsum('bij->bji',input) # output[j][i] = m[i][j]
        # print("x0_shape",x0.get_shape())
        x1 = tf.einsum('bmn,bnk->bmk',input,x0)
        cross = tf.einsum('bmn,nk->bmk',x1,self.W[0]) + self.bias[0] + input
        
        for i in range(1,self.num_layer):
            x0 = tf.einsum('bij->bji',cross) # output[j][i] = m[i][j]
            x1 = tf.einsum('bmn,bnk->bmk',input,x0)
            cross = tf.einsum('bmn,nk->bmk',x1,self.W[i]) + self.bias[i] + cross
        return cross
        
class Deep(tf.keras.layers.Layer):
    def __init__(self,dropout_deep,deep_layer_sizes):
        # input_dim = num_size + embed_size = input_size
        super(Deep, self).__init__()
        self.dropout_deep  = dropout_deep
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


train = pd.read_table('../data/Criteo/train.txt')
train.columns=['label','I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
       'I10', 'I11', 'I12', 'I13','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
       'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
       'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

cont_features=['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
       'I10', 'I11', 'I12', 'I13']
dist_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
       'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
       'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
freq_ = 10
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

# 统计离散特征每个离散值出现的次数组成字典
feat_cnt = Counter()
with open('../data/Criteo/train.txt', 'r') as fin:
    for line_idx, line in enumerate(fin):
        features = line.rstrip('\n').split('\t')
        for idx in categorical_range_:
            if features[idx] == '': continue
            feat_cnt.update([features[idx]])
# Only retain discrete features with high frequency
    fin.close()
dis_feat_set = set() # 高频段的离散字符
for feat, ot in feat_cnt.items():
    if ot >= freq_:
        dis_feat_set.add(feat)


# Create a dictionary for continuous and discrete features
feat_dict = {}
tc = 1
# Continuous features
for idx in continuous_range_:
    feat_dict[idx] = tc
    tc += 1 # 代表占据一列

# Discrete features
cnt_feat_set = set()
with open('../data/Criteo/train.txt', 'r') as fin:
    for line_idx, line in enumerate(fin):
        features = line.rstrip('\n').split('\t')
        for idx in categorical_range_:
            # 排除空字符和低频离散字符
            if features[idx] == '' or features[idx] not in dis_feat_set:
                continue
            # 排除连续性数值
            if features[idx] not in cnt_feat_set:
                cnt_feat_set.add(features[idx])
                # 获取种类数
                feat_dict[features[idx]] = tc
                tc += 1
    fin.close()


train_label = []
train_value = []
train_idx = []

continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)
cont_max_=[]
cont_min_=[]
for cf in cont_features:
    cont_max_.append(max(train[cf]))
    cont_min_.append(min(train[cf]))
cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

def process_line_(line):
    features = line.rstrip('\n').split('\t')
    feat_idx, feat_value, label = [], [], []

    # MinMax Normalization
    for idx in continuous_range_:
        if features[idx] == '':
            feat_idx.append(0)
            feat_value.append(0.0)
        else:
            feat_idx.append(feat_dict[idx])
            # 归一化
            feat_value.append(round((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1], 6))

    # 处理离散型数据
    for idx in categorical_range_:
        if features[idx] == '' or features[idx] not in feat_dict:
            feat_idx.append(0)
            feat_value.append(0.0)
        else:
            feat_idx.append(feat_dict[features[idx]])
            feat_value.append(1.0)
    return feat_idx, feat_value, [int(features[0])]

with open('../data/Criteo/train.txt', 'r') as fin:
    for line_idx, line in enumerate(fin):

        feat_idx, feat_value, label = process_line_(line)
        train_label.append(label)
        train_idx.append(feat_idx)
        train_value.append(feat_value)

    fin.close()
dcn= DCN(num_feat=len(feat_dict) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                deep_layer_sizes=[400, 400])


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_label,train_idx,train_value)).shuffle(10000).batch(32)


@tf.function
def train_one_step(model, optimizer, idx, value, label):
    with tf.GradientTape() as tape:
        output = model(idx,value)
        loss = loss_object(y_true=label, y_pred=output)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 100) for g in grads]
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(label,output)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')

loss_object = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


EPOCHS = 50
for epoch in range(EPOCHS):
    for label, idx, value in train_ds:
        train_one_step(dcn,optimizer,idx, value,label)
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print (template.format(epoch+1,
                             train_loss.result(),train_accuracy.result()))

