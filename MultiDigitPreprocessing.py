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

#contours, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#print(hierarchy)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = contours

"""
親子関係を使って内輪郭を除外する
hierarchy[0][i][0] : Next           : 同じ改装で次の輪郭のインデックス   
hierarchy[0][i][1] : Previous       : 同じ改装で前の輪郭のインデックス
hierarchy[0][i][2] : First Child    : 子の輪郭の最初の子輪郭のインデックス
hierarchy[0][i][3] : Parent         : 子の輪郭の親輪郭のインデックス
"""


"""
filtered_contours = []
for i, cnt in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        filtered_contours.append(cnt)
"""
"""
filtered_contours = []
for i, cnt in enumerate(contours):
    parent = hierarchy[0][i][3]
    if parent == -1:  # 親がいない（外枠）
        filtered_contours.append(cnt)
"""

# ｘ座標でソート
filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

# 枠を見やすくカラーに
img_rectangled = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
#cv2.imshow("rectangle ", img_rectangled)
"""
ここでは枠線は入っていない
"""

# 各数字を切り出す
digits = []
for_model = []
for contour in filtered_contours:
    x,y,w,h = cv2.boundingRect(contour)     # 順番に座標を取り出す
    #if w*h < 200: continue
    area = cv2.contourArea(contour)
    if (area < 100): continue
    print("Area: ", area)
    
    without_rectangle = img_rectangled[y-10:y+h+10,x-10:x+w+10].copy()
    for_model.append(without_rectangle)
    #cv2.imshow('before rectangle', without_rectangle)
    """ 今まで  参照の扱いのせい
    for_model.append(img_rectangled[y-10:y+h+10,x-10:x+w+10]) 
    cv2.imshow('before rectangle', img_rectangled)
    """

    cv2.rectangle(img_rectangled, (x,y), (x+w,y+h), (0,255,0), 2)
    digits.append(img_rectangled[y-10:y+h+10,x-10:x+w+10])   # 配列に部分画像をappend
    #cv2.imshow('b', img_rectangled)
    
"""
for digit in digits:
    cv2.imshow(f"digit: ", digit)
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""


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

    h,w = grayscaled_img.shape
    #print(h,w) # 483, 336
    # TODO: 7が1だと判定されているからもう少し横に伸ばすといいかも？
    resize_to = 24
    if h>w: 
        new_h = resize_to
        new_w = int(w*(new_h / h))
        # 最小幅を保証（特に7のような数字のため）
        border_size=22
        new_w = max(new_w, border_size // 3)

    else:
        new_w = resize_to
        new_h = int(h * (new_w / w))


    # TODO: これをすることで7は1と認識されることがなくなった
    """
    これをすることで7は1と認識されることがなくなった
    """
    # interpolationで点同士の間を補完
    resized_img = cv2.resize(grayscaled_img, (new_w,new_h), interpolation=cv2.INTER_AREA) # 辺が約22で描かれる
    # 補完された部分は薄くなったりするので二値変換
    resized_img[resized_img>=30] = 255
    resized_img[resized_img<30] = 0
    #resized_img = cv2.resize(grayscaled_img,(28,28))
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"Digit {i+1} - Enlarged View")
    plt.axis('off')
    plt.show()
    """


    canvas = np.zeros((28,28), dtype=np.uint8)
    offset_x = (28-new_w) // 2
    offset_y = (28-new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='gray')
    plt.title(f"Digit {i+1} - Enlarged View")
    plt.axis('off')
    plt.show()
    """
    #print(canvas)

    # デバッグ用: 最終的な入力画像を表示
    
    plt.figure(figsize=(3, 3))
    plt.imshow(canvas, cmap='gray')
    plt.title(f"Input to model - Digit")
    plt.show()
    
    
    #resized_img    = cv2.resize(grayscaled_img, (28,28))
    normalized_img = canvas.astype("float32") / 255.0
    input_img      = normalized_img.reshape(1, 28, 28, 1)
    pred = model.predict(input_img)
    pred_class = np.argmax(pred)
    print(f"predicted digit: {pred_class}")
