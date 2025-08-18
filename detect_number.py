"""
認識した数字の輪郭を検出する
"""

"""
このコードの問題点
先にリサイズ
グレースケール画像を渡している
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

threshold = 50
img_array[img_array <= threshold] = 0

# 3. Resize 28x28
img_array = cv2.resize(img_array, (28, 28))
print(img_array)

# img for Canny
img_canny = img_array.astype('uint8')

# 4. 正規化
img_array = img_array.astype("float32") / 255.0

# 5. reshape
preprocessed_img = img_array.reshape(1, 28, 28, 1)


edge = cv2.Canny(img_canny, 50, 150)

contours, hierachy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 一番大きい輪郭を使う
contours = sorted(contours, key=cv2.contourArea, reverse=True)

img_rectangled = img_canny.copy()

x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(img_rectangled, (x,y),(x+w,y+h), 255, 2)


fig, ax = plt.subplots(1, 3, figsize=(20, 20), sharex=True, sharey=True)

ax[0].imshow(img_canny, cmap='gray')
ax[0].set_title('orginal')

ax[1].imshow(edge, cmap='gray')
ax[1].set_title('edged')

ax[2].imshow(img_rectangled, cmap='gray')
ax[2].set_title('rectangled')



ax[0].set_xticks([])
ax[0].set_yticks([])
plt.show()



pred = model.predict(preprocessed_img)
print(pred)

pred_class = np.argmax(pred)
print(pred_class)
