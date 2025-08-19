"""
細い数字（特に「1」）の検出を改良した版
19760 で「1」が検出されない問題を解決
"""
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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
    
    # 水平投影
    h_projection = np.sum(binary, axis=1)
    h_threshold = np.max(h_projection) * 0.05  # より低い閾値
    
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

def detect_thin_characters(img_binary, row_start, row_end):
    """細い文字（特に「1」）の検出に特化"""
    row_img = img_binary[row_start:row_end, :]
    height, width = row_img.shape
    
    # 垂直投影
    v_projection = np.sum(row_img, axis=0)
    
    # デバッグ情報
    print(f"  垂直投影の統計: min={np.min(v_projection)}, max={np.max(v_projection)}, mean={np.mean(v_projection):.1f}")
    
    # 適応的閾値を複数設定
    max_proj = np.max(v_projection)
    
    # 複数の閾値で検出を試行
    thresholds = [
        max_proj * 0.02,  # 非常に低い閾値（細い「1」用）
        max_proj * 0.05,  # 低い閾値
        max_proj * 0.1,   # 標準的な閾値
    ]
    
    all_regions = []
    
    for threshold_ratio, threshold in zip([0.02, 0.05, 0.1], thresholds):
        regions = []
        in_char = False
        start_col = 0
        
        # 最小幅を適応的に設定
        min_char_width = max(2, height // 8)  # より小さな最小幅
        max_char_width = width // 2  # 最大幅制限
        
        print(f"  閾値 {threshold_ratio:.2f} (値: {threshold:.1f}), 最小幅: {min_char_width}")
        
        for i, val in enumerate(v_projection):
            if val > threshold and not in_char:
                start_col = i
                in_char = True
            elif val <= threshold and in_char:
                char_width = i - start_col
                if min_char_width <= char_width <= max_char_width:
                    regions.append((start_col, i, row_start, row_end, threshold_ratio))
                    print(f"    検出: 幅{char_width}, 位置{start_col}-{i}")
                in_char = False
        
        if in_char:
            char_width = len(v_projection) - start_col
            if min_char_width <= char_width <= max_char_width:
                regions.append((start_col, len(v_projection), row_start, row_end, threshold_ratio))
                print(f"    検出(末尾): 幅{char_width}, 位置{start_col}-{len(v_projection)}")
        
        all_regions.extend(regions)
        print(f"    {len(regions)}個の候補を検出")
    
    return all_regions

def merge_and_filter_regions(regions):
    """領域のマージとフィルタリング"""
    if len(regions) <= 1:
        return regions
    
    # x座標でソート
    regions = sorted(regions, key=lambda r: r[0])
    
    # 重複除去（より緩い条件）
    merged = []
    for current in regions:
        merged_flag = False
        
        for i, existing in enumerate(merged):
            # 重複判定（中心点ベース）
            curr_center = (current[0] + current[1]) / 2
            exist_center = (existing[0] + existing[1]) / 2
            
            # より緩い重複判定
            overlap_threshold = min(current[1] - current[0], existing[1] - existing[0]) * 0.3
            
            if abs(curr_center - exist_center) < overlap_threshold:
                # より信頼度の高い方（低い閾値で検出された方）を採用
                if current[4] < existing[4]:  # より低い閾値の方を採用
                    merged[i] = current
                merged_flag = True
                break
        
        if not merged_flag:
            merged.append(current)
    
    # 最終フィルタリング
    filtered = []
    for col_start, col_end, row_start, row_end, threshold_ratio in merged:
        width = col_end - col_start
        height = row_end - row_start
        
        # より緩い条件でフィルタリング
        if (width >= 2 and height >= 10 and width <= height * 2):  # アスペクト比を緩和
            filtered.append((col_start, col_end, row_start, row_end))
    
    return sorted(filtered, key=lambda r: r[0])

def connected_components_backup(img_binary, text_rows):
    """Connected Componentsによるバックアップ検出"""
    backup_regions = []
    
    for row_start, row_end in text_rows:
        roi = img_binary[row_start:row_end, :]
        
        # Connected Components分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        
        for i in range(1, num_labels):  # 0は背景
            x, y, w, h, area = stats[i]
            
            # 細い文字用の緩い条件
            if (w >= 1 and h >= 8 and area >= 10 and
                w <= h * 2 and area >= w * h * 0.2):  # 非常に緩い条件
                
                actual_y = row_start + y
                backup_regions.append((x, x + w, actual_y, actual_y + h))
                print(f"  CC検出: 幅{w}, 高さ{h}, 面積{area}, 位置({x},{actual_y})")
    
    return backup_regions

def extract_digits_improved(img_array):
    """改良された数字抽出（細い文字対応）"""
    binary, text_rows = find_text_regions(img_array)
    
    if not text_rows:
        print("テキスト行が検出されませんでした")
        return [], []
    
    print(f"{len(text_rows)}個のテキスト行を検出")
    
    all_char_regions = []
    
    # 各行で文字を検出
    for i, (row_start, row_end) in enumerate(text_rows):
        print(f"\n=== 行 {i+1} ({row_start}-{row_end}) ===")
        
        # 投影ベースの検出
        char_regions = detect_thin_characters(binary, row_start, row_end)
        print(f"投影ベース検出: {len(char_regions)}個")
        
        # Connected Componentsバックアップ
        cc_regions = connected_components_backup(binary, [(row_start, row_end)])
        print(f"Connected Components検出: {len(cc_regions)}個")
        
        # 両方の結果を統合
        combined_regions = char_regions + [(r[0], r[1], r[2], r[3], 0.0) for r in cc_regions]
        all_char_regions.extend(combined_regions)
    
    # マージとフィルタリング
    final_regions = merge_and_filter_regions(all_char_regions)
    
    print(f"\n最終的に{len(final_regions)}個の文字領域を検出")
    
    # 数字画像を切り出し
    digits = []
    bboxes = []
    
    for col_start, col_end, row_start, row_end in final_regions:
        # 適応的パディング
        width = col_end - col_start
        height = row_end - row_start
        
        # 細い文字には横方向のパディングを多めに
        h_padding = max(3, width // 2) if width < height // 3 else 3
        v_padding = 3
        
        y_start = max(0, row_start - v_padding)
        y_end = min(img_array.shape[0], row_end + v_padding)
        x_start = max(0, col_start - h_padding)
        x_end = min(img_array.shape[1], col_end + h_padding)
        
        digit_roi = img_array[y_start:y_end, x_start:x_end]
        digits.append(digit_roi)
        bboxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
        
        print(f"切り出し {len(digits)}: 元({width}×{height}) → パディング後({x_end-x_start}×{y_end-y_start})")
    
    return digits, bboxes

def predict_digits_enhanced(model, digit_images):
    """細い数字に対応した予測"""
    predictions = []
    confidences = []
    
    for i, digit_img in enumerate(digit_images):
        height, width = digit_img.shape
        
        # 細い文字（1など）の特別処理
        is_thin = width < height / 2
        
        if is_thin:
            print(f"位置 {i+1}: 細い文字として処理 ({width}×{height})")
            
            # 細い文字用の前処理
            # 1. アスペクト比を保持しながら幅を拡張
            target_width = height // 3
            if width < target_width:
                pad_width = (target_width - width) // 2
                digit_img = cv2.copyMakeBorder(digit_img, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
            
            # 2. 画像強調
            digit_img = cv2.morphologyEx(digit_img, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        
        # 28x28リサイズ
        resized = cv2.resize(digit_img, (28, 28))
        
        # 正規化
        normalized = resized.astype("float32") / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)
        
        # 予測
        pred = model.predict(input_img, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        
        predictions.append(pred_class)
        confidences.append(confidence)
        
        print(f"位置 {i+1}: 数字 {pred_class}, 信頼度: {confidence:.3f}")
    
    return predictions, confidences

def visualize_process_detailed(img_original, digits, bboxes, predictions, confidences):
    """詳細な可視化"""
    img_viz = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # バウンディングボックス（細い文字は赤色）
        width = w
        height = h
        is_thin = width < height / 2
        color = (0, 0, 255) if is_thin else (0, 255, 0)  # 細い文字は赤
        
        cv2.rectangle(img_viz, (x, y), (x+w, y+h), color, 2)
        
        # 予測結果
        label = f"{predictions[i]}({confidences[i]:.2f})"
        cv2.putText(img_viz, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 順序番号
        cv2.putText(img_viz, str(i+1), (x+w-15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # サイズ情報
        cv2.putText(img_viz, f"{w}x{h}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return img_viz

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
    
    predictions, confidences = predict_digits_enhanced(model, digits)
    
    # 結果
    result_number = ''.join(map(str, predictions))
    avg_confidence = np.mean(confidences)
    
    print(f"\n=== 最終結果 ===")
    print(f"認識した数字: {result_number}")
    print(f"平均信頼度: {avg_confidence:.3f}")
    
    # 可視化
    img_viz = visualize_process_detailed(img_original, digits, bboxes, predictions, confidences)
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('前処理後')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.title(f'結果: {result_number}')
    plt.axis('off')
    
    # 個別数字表示
    for i, digit in enumerate(digits[:6]):
        plt.subplot(2, 4, 3+i)
        plt.imshow(digit, cmap='gray')
        is_thin = digit.shape[1] < digit.shape[0] / 2
        title_color = 'red' if is_thin else 'black'
        plt.title(f'{i+1}: {predictions[i]} ({confidences[i]:.2f})', color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()