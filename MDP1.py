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
    #print("inverted") 
else: img_array

# 3. 閾値処理
threshold = 30
img_array[img_array <= threshold] = 0


img_binary = img_array.copy()
img_binary[img_binary <= threshold] = 0
img_binary[img_binary > threshold] = 255

# デバッグ用: 二値化結果を確認
"""
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_array, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Binary")
plt.imshow(img_binary, cmap='gray')
plt.show()
"""

"""
輪郭抽出用
"""
img_canny = img_array.astype("uint8")   # このタイプで扱う

# Canny algorithmによるedge detection
edge = cv2.Canny(img_canny, 100, 200)

# edge画像から輪郭(contour)を抽出
"""
6の丸い部分も認識されてしまうのでEXTERNAL -> CCOMP
"""
contours, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
"""
親子関係を使って内輪郭を除外する
hierarchy[0][i][0] : Next           : 同じ改装で次の輪郭のインデックス   
hierarchy[0][i][1] : Previous       : 同じ改装で前の輪郭のインデックス
hierarchy[0][i][2] : First Child    : 子の輪郭の最初の子輪郭のインデックス
hierarchy[0][i][3] : Parent         : 子の輪郭の親輪郭のインデックス
"""
filtered_contours = []
for i, cnt in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        filtered_contours.append(cnt)


# ｘ座標でソート
filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

# 枠を見やすくカラーに
img_rectangled = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
#cv2.imshow("rectangle ", img_rectangled)

# 各数字を切り出す
digits = []
for_model = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if (area < 100 and w < 15 and h < 20): 
        continue

    # 元画像から数字部分を切り出し
    digit_img = img_array[y:y+h, x:x+w]   # ← cv2.boundingRect(contour) の座標を使用
    grayscaled_img = digit_img.astype("uint8")

    # 正方形キャンバスに配置
    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    square[offset_y:offset_y+h, offset_x:offset_x+w] = grayscaled_img

    # 最後に28×28へ
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    resized[resized>=30] = 255
    resized[resized<30] = 0
    
    # デバッグ用: 最終的な入力画像を表示
    plt.figure(figsize=(3, 3))
    plt.imshow(resized, cmap='gray')
    plt.title(f"Input to model - Digit {i+1}")
    plt.show()

    # モデル用に正規化
    normalized_img = resized.astype("float32") / 255.0
    input_img = normalized_img.reshape(1, 28, 28, 1)

    pred = model.predict(input_img)
    pred_class = np.argmax(pred)
    print(f"predicted digit: {pred_class}")

for digit in digits:
    cv2.imshow(f"digit: ", digit)
    cv2.waitKey(0)
cv2.destroyAllWindows()



"""
モデル予測
"""
for digit_img in for_model:
    grayscaled_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grayscaled_img, cmap='gray')
    plt.title(f"Digit {i+1} - Enlarged View")
    plt.axis('off')
    plt.show()
    """
    
    x, y, w, h = cv2.boundingRect(digit_img)
    #h, w = grayscaled_img.shape
    digit_roi = grayscaled_img[y:y+h, x:x+w]
    # 正方形のキャンバスを作る
    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)

    # ROI をキャンバス中央に配置
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    square[offset_y:offset_y+h, offset_x:offset_x+w] = digit_roi

    # 最後に28×28にリサイズ
    resized = cv2.resize(square, (28, 28))



    # デバッグ用: 最終的な入力画像を表示
    plt.figure(figsize=(3, 3))
    plt.imshow(resized, cmap='gray')
    plt.title(f"Input to model - Digit {i+1}")
    plt.show()
    
    #resized_img    = cv2.resize(grayscaled_img, (28,28))
    normalized_img = resized.astype("float32") / 255.0



    input_img      = normalized_img.reshape(1, 28, 28, 1)
    pred = model.predict(input_img)
    pred_class = np.argmax(pred)
    print(f"predicted digit: {pred_class}")
