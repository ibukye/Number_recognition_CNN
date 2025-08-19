"""
認識した数字の輪郭を検出する（改善版）
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
else: 
    img_array = img_array

# 3. 閾値処理
threshold = 30
img_array[img_array <= threshold] = 0

img_binary = img_array.copy()
img_binary[img_binary <= threshold] = 0
img_binary[img_binary > threshold] = 255

# 輪郭抽出用
img_canny = img_array.astype("uint8")

# Canny algorithmによるedge detection
edge = cv2.Canny(img_canny, 100, 200)

# edge画像から輪郭(contour)を抽出
contours, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

print("階層情報:")
print(hierarchy)

# 改善されたフィルタリングロジック
def filter_digit_contours(contours, hierarchy, min_area=300, min_width=15, min_height=20):
    """
    数字の外枠のみを抽出し、内側の円（6、9、0など）を除外する
    
    Args:
        contours: 検出された全輪郭
        hierarchy: 輪郭の階層情報
        min_area: 最小面積
        min_width: 最小幅
        min_height: 最小高さ
    
    Returns:
        filtered_contours: フィルタリングされた輪郭のリスト
    """
    if hierarchy is None:
        return []
    
    valid_contours = []
    
    for i, contour in enumerate(contours):
        # 基本的なサイズフィルタリング
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        if area < min_area or w < min_width or h < min_height:
            continue
        
        # 階層情報を取得
        next_contour = hierarchy[0][i][0]
        prev_contour = hierarchy[0][i][1] 
        first_child = hierarchy[0][i][2]
        parent = hierarchy[0][i][3]
        
        # 外枠の輪郭のみを選択（親がいない、または親が画像の境界）
        if parent == -1:  # 最外枠
            valid_contours.append((i, contour, x, y, w, h, area))
        elif parent != -1:
            # 親がいる場合、親の面積と比較して十分大きければ有効とする
            parent_area = cv2.contourArea(contours[parent])
            area_ratio = area / parent_area if parent_area > 0 else 0
            
            # 親に対して面積比が大きい場合（独立した数字の可能性）
            if area_ratio > 0.3:  # 閾値は調整可能
                valid_contours.append((i, contour, x, y, w, h, area))
    
    # 重複除去：同じような位置・サイズの輪郭を統合
    def is_similar_contour(c1, c2, overlap_threshold=0.5):
        """2つの輪郭が似ているかどうか判定"""
        x1, y1, w1, h1 = c1[2:6]
        x2, y2, w2, h2 = c2[2:6]
        
        # 重複領域を計算
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)
        
        return overlap_area / min_area > overlap_threshold if min_area > 0 else False
    
    # 重複除去処理
    unique_contours = []
    for current in valid_contours:
        is_duplicate = False
        for existing in unique_contours:
            if is_similar_contour(current, existing):
                # より大きな面積の輪郭を採用
                if current[6] > existing[6]:  # area comparison
                    unique_contours.remove(existing)
                    unique_contours.append(current)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_contours.append(current)
    
    # x座標でソート（左から右へ）
    unique_contours.sort(key=lambda x: x[2])  # x座標でソート
    
    return [contour[1] for contour in unique_contours]  # contourオブジェクトのみ返す

# 改善されたフィルタリングを適用
filtered_contours = filter_digit_contours(contours, hierarchy)

print(f"フィルタリング後の輪郭数: {len(filtered_contours)}")

# デバッグ用：検出された輪郭を可視化
img_debug = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
for i, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_debug, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
    cv2.putText(img_debug, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 結果表示
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array, cmap='gray')
plt.subplot(1, 2, 2)
plt.title(f"Detected Digits ({len(filtered_contours)} found)")
plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
plt.show()

# 各数字を切り出してモデルで予測
digits = []
for_model = []

for i, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # パディングを追加して切り出し
    padding = 10
    y_start = max(0, y - padding)
    y_end = min(img_canny.shape[0], y + h + padding)
    x_start = max(0, x - padding)
    x_end = min(img_canny.shape[1], x + w + padding)
    
    digit_region = img_canny[y_start:y_end, x_start:x_end].copy()
    for_model.append(digit_region)

# モデル予測
predicted_digits = []
for i, digit_img in enumerate(for_model):
    print(f"\n=== 数字 {i+1} の処理 ===")
    
    # リサイズ処理の改善
    h, w = digit_img.shape
    resize_to = 24
    
    if h > w:
        new_h = resize_to
        new_w = int(w * (new_h / h))
    else:
        new_w = resize_to
        new_h = int(h * (new_w / w))
    
    # アスペクト比を保持してリサイズ
    resized_img = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 二値化処理
    resized_img[resized_img >= 30] = 255
    resized_img[resized_img < 30] = 0
    
    # 28x28のキャンバスに中央配置
    canvas = np.zeros((28, 28), dtype=np.uint8)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img
    
    # デバッグ用表示
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(digit_img, cmap='gray')
    plt.title(f"Original Digit {i+1}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"Resized ({new_w}x{new_h})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(canvas, cmap='gray')
    plt.title("Final Input (28x28)")
    plt.axis('off')
    plt.show()
    
    # モデル予測
    normalized_img = canvas.astype("float32") / 255.0
    input_img = normalized_img.reshape(1, 28, 28, 1)
    pred = model.predict(input_img, verbose=0)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    
    predicted_digits.append(pred_class)
    print(f"予測結果: {pred_class} (信頼度: {confidence:.3f})")

print(f"\n=== 最終結果 ===")
print(f"検出された数字列: {''.join(map(str, predicted_digits))}")