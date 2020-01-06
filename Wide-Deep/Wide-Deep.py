import tensorflow as tf

import pickle
from util.train_model import train_test_model_demo


class Wide(tf.keras.layers.Layer):
    def __init__(self,units=1):
        # input_dim = num_size + embed_size = input_size
        super(Wide, self).__init__()
#         self.units = units
        self.linear = tf.keras.layers.Dense(units=units,activation='relu')
    def call(self, inputs):
        output = self.linear(inputs)
        return output

class Deep(tf.keras.layers.Layer):
    def __init__(self,num_feat,num_field,dropout_deep,deep_layer_sizes,embedding_size=10):
        # input_dim = num_size + embed_size = input_size
        super(Deep, self).__init__()
        self.num_feat = num_feat # F =features nums
        self.num_field = num_field # N =fields of a feature 
        self.dropout_deep  = dropout_deep
        
        # Embedding 这里采用embeddings层因此大小为F* M F为特征数量，M为embedding的维度
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform') # F * M 
        self.feat_embeddings = feat_embeddings
        
        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        #神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i),tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i),tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i),tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i),tf.keras.layers.Dropout(dropout_deep[i]))
        # last layer
        self.fc = tf.keras.layers.Dense(1,activation=None,use_bias=True)
        
    def call(self,feat_index,feat_value):
        # embedding part  feat_index = inputs为输入 feat_embeddings为一个layer。
        feat_embedding_0 = self.feat_embeddings(feat_index) # Batch * N * M 
#         print(feat_value.get_shape())
        feat_embedding = tf.einsum('bnm,bn->bnm',feat_embedding_0,feat_value)

        y_deep = tf.keras.layers.Flatten()(feat_embedding)
        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self,'dense_' + str(i))(y_deep)
            y_deep = getattr(self,'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self,'activation_' + str(i))(y_deep)
            y_deep = getattr(self,'dropout_' + str(i))(y_deep)
        
        output = self.fc(y_deep)
        return output
    
class WideDeep(tf.keras.Model):
    def __init__(self,num_feat,num_field,dropout_deep,deep_layer_sizes,embedding_size=10):
        super().__init__()
        self.num_feat = num_feat # F =features nums
        self.num_field = num_field # N =fields of a feature 
        self.dropout_deep  = dropout_deep
        
        self.wide = Wide(units=1)
        self.deep = Deep(num_feat,num_field,dropout_deep,deep_layer_sizes)
        self.fc = tf.keras.layers.Dense(1,activation=None,use_bias=True)
        
    def call(self,num_input,feat_index,feat_value):
        x1 = self.wide(num_input)
        x2 = self.deep(feat_index,feat_value)
        
        x3 = tf.keras.layers.concatenate([x1,x2],axis=-1)
        output = self.fc(x3)
        return output
    

if __name__ == '__main__':
    AID_DATA_DIR = "../data/Criteo/"
    feat_dict_ = pickle.load(open(AID_DATA_DIR + '/cross_feat_dict_10.pkl2', 'rb'))

    widedeep = WideDeep(num_feat=len(feat_dict_) + 1, num_field=52, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400],embedding_size=10)

    train_label_path = AID_DATA_DIR + 'traincross_label'
    train_idx_path = AID_DATA_DIR + 'traincross_idx'
    train_value_path = AID_DATA_DIR + 'traincross_value'
    train_num_path = AID_DATA_DIR + 'traincross_num'

    # 这种读取数据方式采用TextLineDataset，数据为大文件时，节省内存，效率训练
    def get_batch_dataset(label_path, idx_path, value_path,num_path):
        label = tf.data.TextLineDataset(label_path)
        idx = tf.data.TextLineDataset(idx_path)
        value = tf.data.TextLineDataset(value_path)
        num = tf.data.TextLineDataset(num_path)

        label = label.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
        idx = idx.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
        value = value.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
        num = num.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)

        batch_dataset = tf.data.Dataset.zip((num,label, idx, value))
        batch_dataset = batch_dataset.shuffle(buffer_size=128)
        batch_dataset = batch_dataset.batch(128)
        batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return batch_dataset
    train_batch_dataset = get_batch_dataset(train_label_path, train_idx_path, train_value_path,train_num_path)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


    @tf.function
    def train_one_step(model, optimizer, idx, value, label, num):
        with tf.GradientTape() as tape:
            output = model(num, idx, value)
            loss = loss_object(y_true=label, y_pred=output)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 100) for g in grads]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, output)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        for num, label, idx, value in train_batch_dataset:
            train_one_step(widedeep, optimizer, idx, value, label,num)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(), train_accuracy.result()))

