from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np

model = load_model("mnist_cnn_with_aug.keras")

root = Tk()
root.withdraw() 
file_path = filedialog.askopenfilename()

img = Image.open(file_path).convert("L")
img_array = np.array(img)

if np.mean(img_array) > 127:
    img_array = 255 - img_array
else: img_array

threshold = 30
img_array[img_array <= threshold] = 0

img_binary = img_array.copy()
img_binary[img_binary <= threshold] = 0
img_binary[img_binary > threshold] = 255

"""
輪郭抽出用
"""
img_canny = img_array.astype("uint8")
edge = cv2.Canny(img_canny, 100, 200)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = contours

filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

img_rectangled = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

digits = []
for_model = []
for contour in filtered_contours:
    x,y,w,h = cv2.boundingRect(contour)   
    area = cv2.contourArea(contour)
    print(area)
    if (area < 100): continue
   
    without_rectangle = img_rectangled[y-10:y+h+10,x-10:x+w+10].copy()
    for_model.append(without_rectangle)
    cv2.rectangle(img_rectangled, (x,y), (x+w,y+h), (0,255,0), 2)
    digits.append(img_rectangled[y-10:y+h+10,x-10:x+w+10])

"""
モデル予測
"""
for digit_img in for_model:
    grayscaled_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    h,w = grayscaled_img.shape
    resize_to = 24
    if h>w: 
        new_h = resize_to
        new_w = int(w*(new_h / h))
        border_size=22
        new_w = max(new_w, border_size // 3)
    else:
        new_w = resize_to
        new_h = int(h * (new_w / w))

    resized_img = cv2.resize(grayscaled_img, (new_w,new_h), interpolation=cv2.INTER_AREA) 
    resized_img[resized_img>=30] = 255
    resized_img[resized_img<30] = 0
   
    canvas = np.zeros((28,28), dtype=np.uint8)
    offset_x = (28-new_w) // 2
    offset_y = (28-new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img

    normalized_img = canvas.astype("float32") / 255.0
    input_img      = normalized_img.reshape(1, 28, 28, 1)
    pred = model.predict(input_img)
    pred_class = np.argmax(pred)
    print(f"predicted digit: {pred_class}")
