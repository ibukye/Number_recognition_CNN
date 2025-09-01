from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("mnist_cnn_with_aug.keras")

def MDP(img_array):
    """
    輪郭抽出用
    """
    img_canny = img_array.astype("uint8")
    edge = cv2.Canny(img_canny, 100, 200)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            filtered_contours.append(cnt)

    filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    img_rectangled = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

    digits = []
    digits_areas = []
    for i in filtered_contours:
        digits_area = cv2.contourArea(i)
        digits_areas.append(digits_area)
    area_mean = np.mean(digits_areas)
    criteria = area_mean/60
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        #print(area) # 6の中の〇をはじくため
        """
        平均をとってその60分の1だった場合ははじく
        この60はもっと小さくてもいいけど一旦
        """
        if (area < criteria or w < 15 or h < 20): 
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

        # モデル用に正規化
        normalized_img = resized.astype("float32") / 255.0
        input_img = normalized_img.reshape(1, 28, 28, 1)

        pred = model.predict(input_img)
        pred_class = np.argmax(pred)
        #print(f"predicted digit: {pred_class}")
        digits.append(str(pred_class))

    number = 0
    for i in range(len(digits)):
        number += int(digits[len(digits) - (i+1)]) * (10 ** i)
    return number
