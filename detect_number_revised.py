"""
認識した数字の輪郭を検出する

"""


from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model("mnist_cnn.keras")

root = Tk()
root.withdraw() 
file_path = filedialog.askopenfilename()

# 1. グレースケール化
img = Image.open(file_path).convert("L")
img_array = np.array(img)

# 2. 白黒反転（MNIST:数字=白(255), 背景=黒(0)）
if np.mean(img_array) > 127:
    img_array = 255 - img_array
    print("inverted") 
else: img_array

# 3. 閾値処理
threshold = 50
img_array[img_array <= threshold] = 0


"""
モデル予測用
"""
img_resized = cv2.resize(img_array, (28,28))
img_normalized = img_resized.astype("float32") / 255.0
preprocessed_img = img_normalized.reshape(1, 28, 28, 1)


"""
輪郭抽出用
"""
img_canny = img_array.astype("uint8")   # このタイプで扱う

# Canny algorithmによるedge detection
"""
img_canny: input img, type: uint8
50 -> 下限の閾値
150 -> 上限の閾値
"""
edge = cv2.Canny(img_canny, 50, 150)

# edge画像から輪郭(contour)を抽出
"""
edge: edge img
cv2.RETR_EXTERNAL: 最外層の輪郭のみ抽出
cv2.CHAIN_APPROX_SIMPLE: 輪郭店を直線で近似してデータ量を削減

contours: 輪郭のリスト
_ : 輪郭の階層構造(heirarchy)
"""
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 一番大きい輪郭を使う
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 枠を見やすくカラーに
"""
色付きの枠を描画するにはカラー画像である必要
1channelのimgを3channel(BGR)に変換
"""
img_rectangled = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

"""
contours[0]に外接する最小の長方形を取得
x,y: 左上の座標
w,h: 幅と高さ
"""
x,y,w,h = cv2.boundingRect(contours[0])

"""
画像上に緑色の矩形を描画
(x,y)       : 左上の座標
(x+w,y+h)   : 右下の座標
(0, 255, 0) : BGRでの色指定
2           : 線の太さ
"""
cv2.rectangle(img_rectangled, (x,y), (x+w,y+h), (0,255,0), 2)

# Visualize
fig, ax = plt.subplots(1, 3, figsize=(20,20), sharex=True, sharey=True)

# 元画像
ax[0].imshow(img_array, cmap="gray")
ax[0].set_title('original')

# edge detected
ax[1].imshow(edge, cmap="gray")
ax[1].set_title('edged')

# rectangled
ax[2].imshow(img_rectangled, cmap="gray")
ax[2].set_title('rectangled')

plt.show()

"""
モデル予測
"""
pred = model.predict(preprocessed_img)
print(pred)

pred_class = np.argmax(pred)
print(pred_class)
