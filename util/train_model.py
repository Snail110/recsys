"""

"""
import tensorflow as tf

def train_test_model_demo(model,train_label_path, train_idx_path, train_value_path):
    # 这种读取数据方式采用TextLineDataset，数据为大文件时，节省内存，效率训练
    def get_batch_dataset(label_path, idx_path, value_path):
        label = tf.data.TextLineDataset(label_path)
        idx = tf.data.TextLineDataset(idx_path)
        value = tf.data.TextLineDataset(value_path)

        label = label.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
        idx = idx.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
        value = value.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)

        batch_dataset = tf.data.Dataset.zip((label, idx, value))
        batch_dataset = batch_dataset.shuffle(buffer_size=128)
        batch_dataset = batch_dataset.batch(128)
        batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return batch_dataset
    train_batch_dataset = get_batch_dataset(train_label_path, train_idx_path, train_value_path)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


    @tf.function
    def train_one_step(model, optimizer, idx, value, label):
        with tf.GradientTape() as tape:
            output = model(idx, value)
            loss = loss_object(y_true=label, y_pred=output)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 100) for g in grads]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, output)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        for label, idx, value in train_batch_dataset:
            train_one_step(model, optimizer, idx, value, label)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(), train_accuracy.result()))

def train_test_model_demo_1(model,train_label, train_idx, train_value):
    # 这种读取数据方式采用tf.data.Dataset.from_tensor_slices，数据为小文件时，便于进行大数据前的调试模型使用。
    def get_dataset(train_label, train_idx, train_value):
        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_label, train_idx, train_value)).shuffle(10000).batch(32)
        return train_ds
    train_batch_dataset = get_dataset(train_label, train_idx, train_value)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')
    # 二分类
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_one_step(model, optimizer, idx, value, label):
        with tf.GradientTape() as tape:
            output = model(idx, value)
            loss = loss_object(y_true=label, y_pred=output)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 100) for g in grads]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, output)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        for label, idx, value in train_batch_dataset:
            train_one_step(model, optimizer, idx, value, label)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(), train_accuracy.result()))