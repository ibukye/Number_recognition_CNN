"""
シンプル版：複数桁数字認識（バウンディングボックス表示のみ）
"""
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path):
    """画像読み込みと前処理"""
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    return img_array

def detect_digits(img_array):
    """数字領域検出"""
    # 二値化
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Connected Components で文字検出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # 数字らしいサイズ・形状でフィルタリング
        if (w >= 5 and h >= 15 and area >= 100 and 
            0.1 <= w/h <= 2.0 and area >= 0.15 * w * h):
            regions.append((x, y, w, h))
    
    # x座標でソート
    return sorted(regions, key=lambda r: r[0])

def extract_and_predict(model, img_array, regions):
    """数字切り出しと予測"""
    predictions = []
    confidences = []
    
    for x, y, w, h in regions:
        # パディング付きで切り出し
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img_array.shape[1], x + w + padding)
        y_end = min(img_array.shape[0], y + h + padding)
        
        digit_roi = img_array[y_start:y_end, x_start:x_end]
        
        # 28x28にリサイズ
        resized = cv2.resize(digit_roi, (28, 28))
        
        # 正規化して予測
        normalized = resized.astype("float32") / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)
        
        pred = model.predict(input_img, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        
        predictions.append(pred_class)
        confidences.append(confidence)
    
    return predictions, confidences

def main():
    # モデル読み込み
    try:
        model = load_model("mnist_cnn.keras")
    except:
        print("mnist_cnn.kerasが見つかりません")
        return
    
    # ファイル選択
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # 処理実行
    img = load_and_preprocess_image(file_path)
    regions = detect_digits(img)
    predictions, confidences = extract_and_predict(model, img, regions)
    
    # 結果表示
    result = ''.join(map(str, predictions))
    print(f"認識結果: {result}")
    
    # バウンディングボックス描画
    img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(regions):
        cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_viz, f"{predictions[i]}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 表示
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.title(f'認識結果: {result}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()