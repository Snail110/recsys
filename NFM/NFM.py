
"""
TensorFlow 2.0 implementation of Product-based Neural Network[1]
Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
[2] Tensorflow implementation of PNN
    https://github.com/Snail110/Awesome-RecSystem-Models/blob/master/Model/PNN_TensorFlow.py
"""
import tensorflow as tf

import pickle
from util.train_model import train_test_model_demo


class BiInteraction(tf.keras.layers.Layer):
    def __init__(self, Units=1, **kwargs):
        self.units = Units
        super(BiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        # self.W = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', trainable=True)
        # self.b = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', trainable=True)
        self.linearlayer = tf.keras.layers.Dense(input_dim, activation='relu', use_bias=True)

    def call(self, input):
        # sum-square-part
        self.summed_features_emb = tf.reduce_sum(input,1) # None * K
        # print("self.summed_features_emb:",self.summed_features_emb.get_shape())
        self.summed_features_emb_square = tf.square(self.summed_features_emb) # None * K
        # square-sum-part
        self.squared_features_emb = tf.square(input)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb,1) # None * K

        # second order
        self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb) # None * K
        print("y_second_order:",self.y_second_order.get_shape()) # 128 * 10
        output = self.linearlayer(self.y_second_order)
        return output

class NFM(tf.keras.Model):
    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, embedding_size=10):
        super().__init__()
        self.num_feat = num_feat  # F =features nums
        self.num_field = num_field  # N =fields of a feature
        self.dropout_deep = dropout_deep

        # Embedding 这里采用embeddings层因此大小为F* M F为特征数量，M为embedding的维度
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size,
                                                    embeddings_initializer='uniform')  # F * M
        self.feat_embeddings = feat_embeddings

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i]))
        self.bilayer = BiInteraction(1)
        # last layer
        self.fc = tf.keras.layers.Dense(1, activation=None, use_bias=True)

        self.linearlayer = tf.keras.layers.Dense(deep_layer_sizes[-1], activation='relu', use_bias=True)

    def call(self, feat_index, feat_value):
        # call函数接收输入变量
        # embedding part  feat_index = inputs为输入 feat_embeddings为一个layer。
        feat_embedding_0 = self.feat_embeddings(feat_index)  # Batch * N * M
        #         print(feat_value.get_shape())
        feat_embedding = tf.einsum('bnm,bn->bnm', feat_embedding_0, feat_value)

        y_deep = self.bilayer(feat_embedding)
        y_linear = self.linearlayer(tf.reduce_sum(feat_embedding,1))

        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self, 'dense_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self, 'activation_' + str(i))(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)
        y = y_deep + y_linear
        output = self.fc(y)

        return output
if __name__ == '__main__':
    AID_DATA_DIR = "../data/Criteo/"
    feat_dict_ = pickle.load(open(AID_DATA_DIR + '/feat_dict_10.pkl2', 'rb'))

    nfm = NFM(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], embedding_size=10)

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train_test_model_demo(nfm,train_label_path, train_idx_path, train_value_path)
