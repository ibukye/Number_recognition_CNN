from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
from MDP_function import MDP

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

threshold = 30
img_array[img_array <= threshold] = 0

img_binary = img_array.copy()
img_binary[img_binary <= threshold] = 0
img_binary[img_binary > threshold] = 255


MDP(img_array)
