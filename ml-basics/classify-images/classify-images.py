# fashion mnist を用いた分類問題

from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow と keras 、そしていくつかの補助ライブラリをインストールします。
import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as plt

print(tf.__version__)  # => 2.0.0-beta1

# fashion mnist のデータセットの持ち込み
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# train_images.shape => 6000 x 28 x 28 : 教師データ(画像)は 6000 枚の 28 x 28 な画像
# train_labels.shape => 60000 : 教師データ(ラベル) は 6000 の数値 (uint8 型)
# test_images .shape =>  10000 x 28 x 28 : テストデータは 10000 の 28 x 28 な画像

# クラスラベル(値) とクラスラベル(名前) の対応付け
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]

# 画像の表示 (省略)

# 訓練画像の正規化 (値域 0-255 の白黒画像から値域 0-1 の白黒画像への変換)
train_images = train_images / 255.0
test_images = test_images / 255.0

# モデルの作成
model = keras.Sequential([
    keras.layers.Flatten(
        input_shape=(28, 28)),  # 28x28 の2次元配列 -> 784=(28x28) の1次元配列への変換
    keras.layers.Dense(128,
                       activation='relu'),  # 全結合層、出力はサイズ128の1次元配列、活性化関数は relu
    keras.layers.Dense(
        10, activation='softmax')  # 全結合層、出力はサイズ10の1次元配列活性化関数は softmax
])

model.compile(
    optimizer='adam',  # 最適化関数 adam
    loss= 'sparse_categorical_crossentropy',  # 損失関数 sparse_categorical_crossentropy
    metrics=['accuracy'],  # メトリクス（精度評価手法） accuracy
)
# optimizer一覧: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers
# loss 一覧: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses
# metrics 一覧: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics

# モデルの学習
model.fit(train_images, train_labels, epochs=5)

# テストデータにおける損失、正答率 (accuracy) の評価
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('[result] Test accuracy {} : Test loss {}'.format(test_acc, test_loss))
# => [result] Test accuracy 0.8745999932289124 : Test loss 0.3492199589967728

# モデルを用いた予測
predictions = model.predict(test_images)

print(predictions[0])
# =>
# [1.9729261e-05 3.7795953e-08 9.0591595e-07 2.6416296e-06 6.2566971e-07
#   2.1061644e-02 1.7808399e-06 6.0303111e-02 4.1502564e-05 9.1856807e-01]

# 正解ラベルの予測
print(np.argmax(predictions[0]))  # => 9
print(class_names[np.argmax(predictions[0])]) #=> Ankle boot
