"""
予測精度を改良した版
17150 → 19760 の認識精度問題を解決
"""
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, segmentation

def load_and_preprocess_image(file_path):
    """画像の読み込みと前処理"""
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    
    mean_val = np.mean(img_array)
    if mean_val > 127:
        img_array = 255 - img_array
        print(f"画像を反転しました (平均値: {mean_val:.1f})")
    
    return img_array

def find_text_regions(img_array):
    """テキスト領域を検出"""
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h_projection = np.sum(binary, axis=1)
    h_threshold = np.max(h_projection) * 0.05
    
    text_rows = []
    in_text = False
    start_row = 0
    
    for i, val in enumerate(h_projection):
        if val > h_threshold and not in_text:
            start_row = i
            in_text = True
        elif val <= h_threshold and in_text:
            text_rows.append((start_row, i))
            in_text = False
    
    if in_text:
        text_rows.append((start_row, len(h_projection)))
    
    return binary, text_rows

def detect_characters_hybrid(img_binary, row_start, row_end):
    """ハイブリッド文字検出（投影 + Connected Components）"""
    row_img = img_binary[row_start:row_end, :]
    height, width = row_img.shape
    
    # 1. Connected Components主体の検出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(row_img, connectivity=8)
    
    cc_regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # より適切なフィルタリング
        if (w >= 3 and h >= 10 and area >= 30 and
            0.1 <= w/h <= 2.0 and  # アスペクト比
            area >= 0.15 * w * h):  # 密度チェック
            
            cc_regions.append((x, x + w, row_start + y, row_start + y + h))
            print(f"  CC検出: 幅{w}, 高さ{h}, 面積{area}, 位置({x},{row_start + y})")
    
    return sorted(cc_regions, key=lambda r: r[0])

def extract_digits_improved(img_array):
    """改良された数字抽出"""
    binary, text_rows = find_text_regions(img_array)
    
    if not text_rows:
        print("テキスト行が検出されませんでした")
        return [], []
    
    print(f"{len(text_rows)}個のテキスト行を検出")
    
    all_char_regions = []
    
    for i, (row_start, row_end) in enumerate(text_rows):
        print(f"\n=== 行 {i+1} ({row_start}-{row_end}) ===")
        char_regions = detect_characters_hybrid(binary, row_start, row_end)
        all_char_regions.extend(char_regions)
        print(f"検出した文字数: {len(char_regions)}")
    
    # 数字画像を切り出し
    digits = []
    bboxes = []
    
    for col_start, col_end, row_start, row_end in all_char_regions:
        # 適応的パディング
        width = col_end - col_start
        height = row_end - row_start
        
        padding = max(5, min(width, height) // 4)
        
        y_start = max(0, row_start - padding)
        y_end = min(img_array.shape[0], row_end + padding)
        x_start = max(0, col_start - padding)
        x_end = min(img_array.shape[1], col_end + padding)
        
        digit_roi = img_array[y_start:y_end, x_start:x_end]
        digits.append(digit_roi)
        bboxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
    
    print(f"\n最終的に{len(digits)}個の文字を抽出")
    return digits, bboxes

def preprocess_digit_for_prediction(digit_img):
    """予測用の高品質前処理"""
    
    # 1. ノイズ除去
    digit_img = cv2.medianBlur(digit_img, 3)
    
    # 2. コントラスト調整（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    digit_img = clahe.apply(digit_img)
    
    # 3. 二値化（複数手法を試して最適選択）
    _, binary1 = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY)
    
    # エッジの保存度で判定
    edges1 = cv2.Canny(binary1, 50, 150)
    edges2 = cv2.Canny(binary2, 50, 150)
    
    if np.sum(edges1) > np.sum(edges2):
        binary = binary1
    else:
        binary = binary2
    
    # 4. モルフォロジー処理
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1,1), np.uint8))
    
    return binary

def resize_with_aspect_ratio(img, target_size=28):
    """アスペクト比を保持してリサイズ"""
    h, w = img.shape
    
    # アスペクト比を保持
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    
    # リサイズ
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # パディングでtarget_size x target_sizeにする
    result = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 中央に配置
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

def predict_with_ensemble(model, digit_img):
    """アンサンブル予測（複数の前処理手法）"""
    
    predictions = []
    confidences = []
    
    # 手法1: 標準的な前処理
    processed1 = preprocess_digit_for_prediction(digit_img.copy())
    resized1 = resize_with_aspect_ratio(processed1, 28)
    normalized1 = resized1.astype("float32") / 255.0
    input1 = normalized1.reshape(1, 28, 28, 1)
    pred1 = model.predict(input1, verbose=0)
    
    # 手法2: 直接リサイズ（前処理最小限）
    resized2 = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_CUBIC)
    normalized2 = resized2.astype("float32") / 255.0
    input2 = normalized2.reshape(1, 28, 28, 1)
    pred2 = model.predict(input2, verbose=0)
    
    # 手法3: ガウシアンブラー + シャープニング
    blurred = cv2.GaussianBlur(digit_img, (3, 3), 0)
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel_sharpen)
    resized3 = resize_with_aspect_ratio(sharpened, 28)
    normalized3 = resized3.astype("float32") / 255.0
    input3 = normalized3.reshape(1, 28, 28, 1)
    pred3 = model.predict(input3, verbose=0)
    
    # 手法4: 侵食・膨張処理
    kernel = np.ones((2,2), np.uint8)
    eroded = cv2.erode(digit_img, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    resized4 = resize_with_aspect_ratio(dilated, 28)
    normalized4 = resized4.astype("float32") / 255.0
    input4 = normalized4.reshape(1, 28, 28, 1)
    pred4 = model.predict(input4, verbose=0)
    
    # アンサンブル（重み付き平均）
    ensemble_pred = (pred1 * 0.4 + pred2 * 0.3 + pred3 * 0.2 + pred4 * 0.1)
    
    pred_class = np.argmax(ensemble_pred)
    confidence = np.max(ensemble_pred)
    
    # 個別結果も記録
    individual_preds = [
        (np.argmax(pred1), np.max(pred1), "前処理強化"),
        (np.argmax(pred2), np.max(pred2), "直接リサイズ"),
        (np.argmax(pred3), np.max(pred3), "シャープニング"),
        (np.argmax(pred4), np.max(pred4), "モルフォロジー")
    ]
    
    return pred_class, confidence, individual_preds, [resized1, resized2, resized3, resized4]

def predict_digits_enhanced(model, digit_images):
    """強化された予測"""
    predictions = []
    confidences = []
    all_individual_results = []
    all_processed_images = []
    
    for i, digit_img in enumerate(digit_images):
        print(f"\n=== 位置 {i+1} ===")
        
        pred_class, confidence, individual_preds, processed_imgs = predict_with_ensemble(model, digit_img)
        
        predictions.append(pred_class)
        confidences.append(confidence)
        all_individual_results.append(individual_preds)
        all_processed_images.append(processed_imgs)
        
        print(f"アンサンブル結果: 数字 {pred_class}, 信頼度: {confidence:.3f}")
        print("個別結果:")
        for pred, conf, method in individual_preds:
            print(f"  {method}: {pred} ({conf:.3f})")
        
        # 信頼度が低い場合の警告
        if confidence < 0.7:
            print(f"⚠️  低信頼度: {confidence:.3f}")
    
    return predictions, confidences, all_individual_results, all_processed_images

def visualize_prediction_process(original_digits, processed_images_list, predictions, individual_results):
    """予測過程の詳細可視化"""
    n_digits = len(original_digits)
    
    fig, axes = plt.subplots(n_digits, 6, figsize=(18, 3*n_digits))
    if n_digits == 1:
        axes = axes.reshape(1, -1)
    
    methods = ["元画像", "前処理強化", "直接リサイズ", "シャープニング", "モルフォロジー", "結果"]
    
    for i in range(n_digits):
        # 元画像
        axes[i, 0].imshow(original_digits[i], cmap='gray')
        axes[i, 0].set_title(f"位置{i+1}: 元画像")
        axes[i, 0].axis('off')
        
        # 各処理結果
        for j, processed_img in enumerate(processed_images_list[i]):
            axes[i, j+1].imshow(processed_img, cmap='gray')
            pred, conf, method = individual_results[i][j]
            axes[i, j+1].set_title(f"{method}\n{pred}({conf:.2f})")
            axes[i, j+1].axis('off')
        
        # 最終結果
        axes[i, 5].text(0.5, 0.5, f"最終結果:\n{predictions[i]}", 
                       ha='center', va='center', fontsize=20, fontweight='bold')
        axes[i, 5].set_title("アンサンブル")
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        model = load_model("mnist_cnn.keras")
        print("MNISTモデルを読み込みました")
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return
    
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    
    if not file_path:
        return
    
    print(f"処理中: {file_path}")
    
    # 処理実行
    img_original = load_and_preprocess_image(file_path)
    digits, bboxes = extract_digits_improved(img_original)
    
    if not digits:
        print("数字が検出されませんでした")
        return
    
    predictions, confidences, individual_results, processed_images_list = predict_digits_enhanced(model, digits)
    
    # 結果
    result_number = ''.join(map(str, predictions))
    avg_confidence = np.mean(confidences)
    
    print(f"\n" + "="*50)
    print(f"最終結果: {result_number}")
    print(f"平均信頼度: {avg_confidence:.3f}")
    print("="*50)
    
    # 可視化
    visualize_prediction_process(digits, processed_images_list, predictions, individual_results)
    
    # バウンディングボックス表示
    img_viz = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(bboxes):
        color = (0, 255, 0) if confidences[i] >= 0.7 else (0, 165, 255)  # 低信頼度はオレンジ
        cv2.rectangle(img_viz, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_viz, f"{predictions[i]}({confidences[i]:.2f})", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.title(f'最終結果: {result_number}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()