"""
改良版：輪郭検出による複数桁数字認識
"""
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path):
    """画像の読み込みと前処理"""
    # 1. グレースケール化
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    
    # 2. 白黒反転チェック（改良版）
    mean_val = np.mean(img_array)
    if mean_val > 127:
        img_array = 255 - img_array
        print(f"画像を反転しました (平均値: {mean_val:.1f})")
    
    # 3. ノイズ除去（ガウシアンフィルタ）
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # 4. 閾値処理（OTSU自動閾値も試行）
    threshold = 50
    _, binary = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
    
    # OTSU法も試す
    _, binary_otsu = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_array, binary, binary_otsu

def find_digit_contours(img_array, min_area=300, min_width=10, min_height=15):
    """輪郭検出と数字領域の抽出"""
    img_canny = img_array.astype("uint8")
    
    # Cannyエッジ検出
    edge = cv2.Canny(img_canny, 50, 150, apertureSize=3)
    
    # モルフォロジー処理でエッジを強化
    kernel = np.ones((2, 2), np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # 親輪郭のみを抽出（内輪郭を除外）
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # 親輪郭
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # フィルタリング条件を改良
            aspect_ratio = h / w if w > 0 else 0
            if (area >= min_area and 
                w >= min_width and h >= min_height and
                0.5 <= aspect_ratio <= 3.0):  # 数字らしいアスペクト比
                filtered_contours.append(cnt)
    
    # x座標でソート
    filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
    return filtered_contours, edge

def extract_digits(img_array, contours, padding=10):
    """数字領域の切り出し"""
    img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    digits = []
    bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # バウンディングボックスを描画
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # パディングを追加して切り出し
        y_start = max(0, y - padding)
        y_end = min(img_array.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img_array.shape[1], x + w + padding)
        
        digit_roi = img_array[y_start:y_end, x_start:x_end]
        digits.append(digit_roi)
        bboxes.append((x, y, w, h))
    
    return digits, bboxes, img_color

def predict_digits(model, digit_images):
    """各数字の予測"""
    predictions = []
    confidences = []
    
    for digit_img in digit_images:
        # 28x28にリサイズ
        resized_img = cv2.resize(digit_img, (28, 28))
        
        # 正規化
        normalized_img = resized_img.astype("float32") / 255.0
        
        # バッチ次元を追加
        input_img = normalized_img.reshape(1, 28, 28, 1)
        
        # 予測
        pred = model.predict(input_img, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        
        predictions.append(pred_class)
        confidences.append(confidence)
        
        print(f"予測数字: {pred_class}, 信頼度: {confidence:.3f}")
    
    return predictions, confidences

def main():
    # モデル読み込み
    try:
        model = load_model("mnist_cnn.keras")
    except:
        print("モデルファイルが見つかりません。")
        return
    
    # ファイル選択
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    
    if not file_path:
        print("ファイルが選択されませんでした。")
        return
    
    # 画像処理
    img_original, img_binary, img_binary_otsu = load_and_preprocess_image(file_path)
    
    # 輪郭検出
    contours, edge_img = find_digit_contours(img_binary)
    
    if not contours:
        print("数字が検出されませんでした。")
        return
    
    print(f"{len(contours)}個の数字候補を検出しました。")
    
    # 数字領域の切り出し
    digits, bboxes, img_with_boxes = extract_digits(img_binary, contours)
    
    # 予測実行
    predictions, confidences = predict_digits(model, digits)
    
    # 結果表示
    result_number = ''.join(map(str, predictions))
    avg_confidence = np.mean(confidences)
    
    print(f"\n=== 認識結果 ===")
    print(f"認識した数字: {result_number}")
    print(f"平均信頼度: {avg_confidence:.3f}")
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('元画像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(img_binary, cmap='gray')
    plt.title('二値化画像')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(edge_img, cmap='gray')
    plt.title('エッジ検出')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f'検出結果: {result_number}')
    plt.axis('off')
    
    # 個別数字表示
    plt.subplot(2, 3, 5)
    if digits:
        combined_digits = np.hstack([cv2.resize(d, (28, 28)) for d in digits[:5]])
        plt.imshow(combined_digits, cmap='gray')
        plt.title('切り出した数字（最大5個）')
        plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.bar(range(len(confidences)), confidences)
    plt.title('各数字の信頼度')
    plt.xlabel('数字の位置')
    plt.ylabel('信頼度')
    plt.xticks(range(len(predictions)), predictions)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()