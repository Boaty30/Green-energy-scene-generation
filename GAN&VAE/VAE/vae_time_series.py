import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

# 使用GPU进行训练
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

# 加载数据集
data = np.load('ca_data_99solar_15min.npy')  # 假设数据存储在这个文件中
data = data.reshape(-1, 96, 1)  # 调整数据形状

# 数据归一化
data_min = data.min()
data_max = data.max()
data = (data - data_min) / (data_max - data_min)

# 超参数
batch_size = 256  # 增大批量大小
original_dim = 96
latent_dim = 10
intermediate_dim = 128
epochs = 500  # 增加epochs数量
epsilon_std = 1.0

# 构建编码器
x = Input(shape=(original_dim, 1))
h = LSTM(intermediate_dim, return_sequences=False)(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 采样层
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 构建解码器
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(original_dim * 1, activation='relu')
decoder_reshape = Reshape((original_dim, 1))
decoder_mean = LSTM(1, return_sequences=True, activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
x_decoded_mean = decoder_mean(reshape_decoded)

# 自定义损失层
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='adam')
vae.summary()

# 提前加载数据到内存中，减少数据传输时间
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

# 使用TensorFlow的数据管道优化数据加载
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 训练VAE并保存loss值
history = vae.fit(train_dataset,
                  epochs=epochs,
                  validation_data=val_dataset)

# 保存模型
vae.save('vae_time_series.h5')

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_x_decoded_mean = decoder_mean(_reshape_decoded)
generator = Model(decoder_input, _x_decoded_mean)
generator.save('generator_time_series.h5')

# 生成样本并反归一化
z_sample = np.random.normal(size=(1, latent_dim))
generated_sample = generator.predict(z_sample)
generated_sample = generated_sample * (data_max - data_min) + data_min

# 打印生成的样本
print(generated_sample)

# 绘制训练过程的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()

# 绘制生成的样本
plt.figure(figsize=(10, 5))
plt.plot(generated_sample.flatten(), label='Generated Sample')
plt.title('Generated Sample')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.savefig('generated_sample.png')
plt.show()
