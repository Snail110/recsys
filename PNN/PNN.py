
import tensorflow as tf

import pickle
from util.train_model import train_test_model_demo
class PNN(tf.keras.Model):
    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10, reg_l1=0.01,
                 reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat  # F =features nums
        self.num_field = num_field  # N =fields of a feature
        self.product_layer_dim = product_layer_dim  # D1 pnn dim
        self.dropout_deep = dropout_deep

        # Embedding 这里采用embeddings层因此大小为F* M F为特征数量，M为embedding的维度
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size,
                                                    embeddings_initializer='uniform')  # F * M
        self.feat_embeddings = feat_embeddings

        # 定义随机初始化
        initializer = tf.initializers.GlorotUniform()

        # linear part 线性层就是embedding层的复制，因此线性信号权重大小是D1 * N * M，为什么因此是线性层维度为 D1，embedding层维度为N* M
        # 因此权重大小为D1 * N *M
        self.linear_weights = tf.Variable(
            initializer(shape=(product_layer_dim, num_field, embedding_size)))  # D1 * N * M

        # quadratic part
        self.product_type = product_type
        if product_type == 'inner':
            self.theta = tf.Variable(initializer(shape=(product_layer_dim, num_field)))  # D1 * N

        else:
            self.quadratic_weights = tf.Variable(
                initializer(shape=(product_layer_dim, embedding_size, embedding_size)))  # D1 * M * M

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i]))

        # last layer
        self.fc = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def call(self, feat_index, feat_value):
        # call函数接收输入变量
        # embedding part  feat_index = inputs为输入 feat_embeddings为一个layer。
        feat_embedding_0 = self.feat_embeddings(feat_index)  # Batch * N * M
        #         print(feat_value.get_shape())
        feat_embedding = tf.einsum('bnm,bn->bnm', feat_embedding_0, feat_value)
        # linear part
        lz = tf.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = tf.einsum('bnm,dn->bdnm', feat_embedding, self.theta)  # Batch * D1 * N * M
            lp = tf.einsum('bdnm,bdnm->bd', theta, theta)  # Batch * D1
        else:
            embed_sum = tf.reduce_sum(feat_embedding, axis=1)  # Batch * M
            p = tf.einsum('bm,bn->bmn', embed_sum, embed_sum)
            lp = tf.einsum('bmn,dmn->bd', p, self.quadratic_weights)  # Batch * D1

        y_deep = tf.concat((lz, lp), axis=1)
        y_deep = tf.keras.layers.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self, 'dense_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self, 'activation_' + str(i))(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        output = self.fc(y_deep)

        return output
if __name__ == '__main__':
    AID_DATA_DIR = "../data/Criteo/"
    feat_dict_ = pickle.load(open(AID_DATA_DIR + '/feat_dict_10.pkl2', 'rb'))

    pnn = PNN(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer')

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train_test_model_demo(pnn,train_label_path, train_idx_path, train_value_path)
