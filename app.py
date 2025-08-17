# import the mnist dataset
from tensorflow.keras.datasets import mnist


# load and split (unpack)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
print(type(x_train))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
"""


# Preprocessing

### Normalization
# pixel = [0, 255]
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

### Resahpe
# CNNの場合、Keras/Tensorflowは入力を(num_sample, height, width, num_channel)
# MNISTはgray scaleなのでnum_channelは1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

### One-hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


### Construct a model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # -- Block 1 --
    # Conv2D -> 小さなkernelで局所的な特徴を検出
    # Relu -> 非線形性を加えて表現力をアップ
    # MaxPooling2D -> 特徴マップをダウンサンプリングして「位置に頑健な特徴に」
    Conv2D(filters=16, kernel_size=(3, 3), padding="same", input_shape=(28, 28, 1)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # -- Block 2 --
    # もう一度 畳み込み → ReLU → プーリング
    #これで「より抽象的な特徴」を抜き出す。
    #フィルタ数は Block1 より多くするのが一般的
    Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=(28, 28, 1)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # 2次元(height x width x channel)の特徴マップを1次元ベクトルに変換
    # ここからは全結合層ネットワークと同じ扱い
    Flatten(),

    # -- Classifier --
    # Dense(全結合層) + ReLU -> ここで「抽出した特徴を組み合わせて分類」
    Dense(units=64, activation="relu"),
    # Dropout -> Neuronをランダムに無効化してoverfittingを防ぐ
    Dropout(rate=0.3),

    # 最終Dense (unit_number = 10) + softmax -> 10クラスの確率分布を出力
    Dense(10),
    Activation("softmax"),

])

#model.summary()


### Compile
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


### fit (learning)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=2,
)



### Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('test_acc: ', test_acc)






### Debug (where did it make mistake)
"""
import numpy as np

y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

mis_idx = np.where(y_pred != y_true)[0]
print(mis_idx)
"""

model.save("mnist_cnn.keras")
# 再読み込み: tf.keras.models.load_model("mnist_cnn.keras")
