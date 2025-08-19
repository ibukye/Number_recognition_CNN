import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

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
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,        # 軽微な回転のみ
    width_shift_range=0.1,    # 軽微なシフト
    height_shift_range=0.1,
    zoom_range=0.1,          # 軽微なズーム
    shear_range=0.1,         # 軽微なせん断変形
    horizontal_flip=False,    # 数字では水平反転しない
    vertical_flip=False,      # 数字では垂直反転しない
    fill_mode='nearest'       # 変形時の空白埋め方法
)

datagen.fit(x_train)

### Construct a model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

# モデルの構築（CNNアーキテクチャ）
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# データ拡張なしでの訓練（比較用）
print("=== データ拡張なしでの訓練 ===")
history_no_aug = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,  # 実際は10-20エポック推奨
    batch_size=128,
    verbose=1
)

# モデルを一旦保存
model.save("mnist_cnn_no_aug.keras")




# 新しいモデルを作成（データ拡張あり）
model_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_aug.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# データ拡張ありでの訓練
print("\n=== データ拡張ありでの訓練 ===")
history_aug = model_aug.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    steps_per_epoch=len(x_train) // 128,
    validation_data=(x_test, y_test),
    epochs=5,  # 実際は10-20エポック推奨
    verbose=1
)

# データ拡張ありのモデルを保存
model_aug.save("mnist_cnn_with_aug.keras")

# 結果の比較
print("\n=== 結果比較 ===")
score_no_aug = model.evaluate(x_test, y_test, verbose=0)
score_aug = model_aug.evaluate(x_test, y_test, verbose=0)

print(f"データ拡張なし - Loss: {score_no_aug[0]:.4f}, Accuracy: {score_no_aug[1]:.4f}")
print(f"データ拡張あり - Loss: {score_aug[0]:.4f}, Accuracy: {score_aug[1]:.4f}")

# 学習曲線の可視化
plt.figure(figsize=(15, 5))

# 精度の比較
plt.subplot(1, 3, 1)
plt.plot(history_no_aug.history['accuracy'], label='訓練（拡張なし）')
plt.plot(history_no_aug.history['val_accuracy'], label='検証（拡張なし）')
plt.plot(history_aug.history['accuracy'], label='訓練（拡張あり）')
plt.plot(history_aug.history['val_accuracy'], label='検証（拡張あり）')
plt.title('モデル精度の比較')
plt.ylabel('精度')
plt.xlabel('エポック')
plt.legend()

# 損失の比較
plt.subplot(1, 3, 2)
plt.plot(history_no_aug.history['loss'], label='訓練（拡張なし）')
plt.plot(history_no_aug.history['val_loss'], label='検証（拡張なし）')
plt.plot(history_aug.history['loss'], label='訓練（拡張あり）')
plt.plot(history_aug.history['val_loss'], label='検証（拡張あり）')
plt.title('モデル損失の比較')
plt.ylabel('損失')
plt.xlabel('エポック')
plt.legend()

# データ拡張の例を表示
plt.subplot(1, 3, 3)
sample_idx = 0
original_img = x_train[sample_idx:sample_idx+1]
augmented_imgs = []

aug_iter = datagen.flow(original_img, batch_size=1)
for i in range(9):
    augmented_imgs.append(next(aug_iter)[0])

# 3x3グリッドで表示
for i in range(9):
    plt.subplot(3, 3, i+1)
    if i == 0:
        plt.imshow(original_img[0,:,:,0], cmap='gray')
        plt.title('元画像')
    else:
        plt.imshow(augmented_imgs[i-1][:,:,0], cmap='gray')
        plt.title(f'拡張{i}')
    plt.axis('off')

plt.suptitle('データ拡張の例')
plt.tight_layout()
plt.show()

# 各数字でのテスト（特に7と1の区別）
print("\n=== 7と1の予測性能比較 ===")

# テストデータから7と1のみを抽出
indices_7 = np.where(np.argmax(y_test, axis=1) == 7)[0][:100]
indices_1 = np.where(np.argmax(y_test, axis=1) == 1)[0][:100]

X_test_7 = x_test[indices_7]
X_test_1 = x_test[indices_1]

# 両モデルでの予測
pred_no_aug_7 = model.predict(X_test_7, verbose=0)
pred_aug_7 = model_aug.predict(X_test_7, verbose=0)
pred_no_aug_1 = model.predict(X_test_1, verbose=0)
pred_aug_1 = model_aug.predict(X_test_1, verbose=0)

# 7の正解率
acc_no_aug_7 = np.mean(np.argmax(pred_no_aug_7, axis=1) == 7)
acc_aug_7 = np.mean(np.argmax(pred_aug_7, axis=1) == 7)

# 1の正解率
acc_no_aug_1 = np.mean(np.argmax(pred_no_aug_1, axis=1) == 1)
acc_aug_1 = np.mean(np.argmax(pred_aug_1, axis=1) == 1)

print(f"7の正解率 - 拡張なし: {acc_no_aug_7:.3f}, 拡張あり: {acc_aug_7:.3f}")
print(f"1の正解率 - 拡張なし: {acc_no_aug_1:.3f}, 拡張あり: {acc_aug_1:.3f}")

print("\n訓練完了！新しいモデルファイル:")
print("- mnist_cnn_no_aug.keras (拡張なし)")
print("- mnist_cnn_with_aug.keras (拡張あり)")


