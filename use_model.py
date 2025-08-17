from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np

model = load_model("mnist_cnn.keras")

# ダイアログを表示してファイル選択
root = Tk()
root.withdraw()  # Tkウィンドウを表示しない
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

# 4. 正規化
img_array = img_array.astype("float32") / 255.0

# 5. reshape
preprocessed_img = img_array.reshape(1, 28, 28, 1)


pred = model.predict(preprocessed_img)
print(pred)

pred_class = np.argmax(pred)
print(pred_class)
