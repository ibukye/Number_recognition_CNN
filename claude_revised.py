"""
改良版：輪郭検出による複数桁数字認識
19760 → 7315283 の問題を解決
"""
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path):
    """画像の読み込みと前処理"""
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    
    # 白黒反転チェック
    mean_val = np.mean(img_array)
    if mean_val > 127:
        img_array = 255 - img_array
        print(f"画像を反転しました (平均値: {mean_val:.1f})")
    
    return img_array

def find_text_regions(img_array):
    """テキスト領域を検出してROIを決定"""
    # 二値化
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 水平方向の投影（行の検出）
    h_projection = np.sum(binary, axis=1)
    h_threshold = np.max(h_projection) * 0.1
    
    # テキスト行の上下境界を見つける
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
    
    if in_text:  # 最後まで文字が続いている場合
        text_rows.append((start_row, len(h_projection)))
    
    return binary, text_rows

def segment_characters_in_row(img_binary, row_start, row_end):
    """1行の中で文字を分割"""
    row_img = img_binary[row_start:row_end, :]
    
    # 垂直方向の投影（文字の分離）
    v_projection = np.sum(row_img, axis=0)
    
    # ノイズ除去のためのスムージング
    kernel_size = max(1, (row_end - row_start) // 10)
    kernel = np.ones(kernel_size) / kernel_size
    v_projection_smooth = np.convolve(v_projection, kernel, mode='same')
    
    # 動的閾値の設定
    v_threshold = np.max(v_projection_smooth) * 0.05
    
    # 文字境界の検出
    char_regions = []
    in_char = False
    start_col = 0
    min_char_width = (row_end - row_start) // 4  # 文字の最小幅
    
    for i, val in enumerate(v_projection_smooth):
        if val > v_threshold and not in_char:
            start_col = i
            in_char = True
        elif val <= v_threshold and in_char:
            char_width = i - start_col
            if char_width >= min_char_width:  # 最小幅チェック
                char_regions.append((start_col, i, row_start, row_end))
            in_char = False
    
    if in_char:  # 最後まで文字が続いている場合
        char_width = len(v_projection_smooth) - start_col
        if char_width >= min_char_width:
            char_regions.append((start_col, len(v_projection_smooth), row_start, row_end))
    
    return char_regions

def merge_overlapping_regions(regions, overlap_threshold=0.3):
    """重複する領域をマージ"""
    if len(regions) <= 1:
        return regions
    
    # x座標でソート
    regions = sorted(regions, key=lambda r: r[0])
    merged = [regions[0]]
    
    for current in regions[1:]:
        last = merged[-1]
        
        # 重複チェック
        overlap_start = max(last[0], current[0])
        overlap_end = min(last[1], current[1])
        overlap_width = max(0, overlap_end - overlap_start)
        
        min_width = min(last[1] - last[0], current[1] - current[0])
        overlap_ratio = overlap_width / min_width if min_width > 0 else 0
        
        if overlap_ratio > overlap_threshold:
            # マージ
            merged[-1] = (
                min(last[0], current[0]),
                max(last[1], current[1]),
                min(last[2], current[2]),
                max(last[3], current[3])
            )
        else:
            merged.append(current)
    
    return merged

def extract_digits_improved(img_array):
    """改良された数字抽出"""
    binary, text_rows = find_text_regions(img_array)
    
    if not text_rows:
        print("テキスト行が検出されませんでした")
        return [], []
    
    print(f"{len(text_rows)}個のテキスト行を検出")
    
    all_char_regions = []
    
    # 各行で文字を検出
    for row_start, row_end in text_rows:
        char_regions = segment_characters_in_row(binary, row_start, row_end)
        all_char_regions.extend(char_regions)
        print(f"行 {row_start}-{row_end}: {len(char_regions)}個の文字を検出")
    
    # 重複領域をマージ
    all_char_regions = merge_overlapping_regions(all_char_regions)
    
    # 面積による追加フィルタリング
    filtered_regions = []
    for col_start, col_end, row_start, row_end in all_char_regions:
        width = col_end - col_start
        height = row_end - row_start
        area = width * height
        
        # 数字らしいサイズ・アスペクト比チェック
        if (width >= 10 and height >= 15 and area >= 150 and
            0.3 <= width/height <= 2.0):
            filtered_regions.append((col_start, col_end, row_start, row_end))
    
    # x座標で最終ソート
    filtered_regions = sorted(filtered_regions, key=lambda r: r[0])
    
    # 数字画像を切り出し
    digits = []
    bboxes = []
    
    for col_start, col_end, row_start, row_end in filtered_regions:
        # パディング追加
        padding = 5
        y_start = max(0, row_start - padding)
        y_end = min(img_array.shape[0], row_end + padding)
        x_start = max(0, col_start - padding)
        x_end = min(img_array.shape[1], col_end + padding)
        
        digit_roi = img_array[y_start:y_end, x_start:x_end]
        digits.append(digit_roi)
        bboxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
    
    return digits, bboxes

def predict_digits_with_postprocessing(model, digit_images):
    """予測結果の後処理付き"""
    predictions = []
    confidences = []
    
    for i, digit_img in enumerate(digit_images):
        # 複数のスケールで予測して最も信頼度の高いものを選択
        scales = [28, 32, 24]  # 異なるサイズで試行
        best_pred = None
        best_conf = 0
        
        for scale in scales:
            resized = cv2.resize(digit_img, (scale, scale))
            if scale != 28:
                # 28x28にクロップまたはパディング
                if scale > 28:
                    # 中央クロップ
                    start = (scale - 28) // 2
                    resized = resized[start:start+28, start:start+28]
                else:
                    # パディング
                    pad = (28 - scale) // 2
                    resized = cv2.copyMakeBorder(resized, pad, 28-scale-pad, pad, 28-scale-pad, cv2.BORDER_CONSTANT, value=0)
            
            normalized = resized.astype("float32") / 255.0
            input_img = normalized.reshape(1, 28, 28, 1)
            
            pred = model.predict(input_img, verbose=0)
            pred_class = np.argmax(pred)
            confidence = np.max(pred)
            
            if confidence > best_conf:
                best_pred = pred_class
                best_conf = confidence
        
        predictions.append(best_pred)
        confidences.append(best_conf)
        
        print(f"位置 {i+1}: 数字 {best_pred}, 信頼度: {best_conf:.3f}")
    
    return predictions, confidences

def visualize_process(img_original, digits, bboxes, predictions, confidences):
    """処理過程の可視化"""
    # バウンディングボックスを描画
    img_viz = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # バウンディングボックス
        cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 予測結果と信頼度
        label = f"{predictions[i]}({confidences[i]:.2f})"
        cv2.putText(img_viz, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 順序番号
        cv2.putText(img_viz, str(i+1), (x+w-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return img_viz

def main():
    # モデル読み込み
    try:
        model = load_model("mnist_cnn.keras")
        print("MNISTモデルを読み込みました")
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return
    
    # ファイル選択
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    
    if not file_path:
        print("ファイルが選択されませんでした")
        return
    
    print(f"処理中: {file_path}")
    
    # 画像処理
    img_original = load_and_preprocess_image(file_path)
    
    # 改良された数字抽出
    digits, bboxes = extract_digits_improved(img_original)
    
    if not digits:
        print("数字が検出されませんでした")
        return
    
    print(f"\n{len(digits)}個の文字領域を検出しました")
    
    # 予測実行
    predictions, confidences = predict_digits_with_postprocessing(model, digits)
    
    # 結果
    result_number = ''.join(map(str, predictions))
    avg_confidence = np.mean(confidences)
    
    print(f"\n=== 最終結果 ===")
    print(f"認識した数字: {result_number}")
    print(f"平均信頼度: {avg_confidence:.3f}")
    
    # 可視化
    img_viz = visualize_process(img_original, digits, bboxes, predictions, confidences)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('前処理後の画像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.title(f'検出結果: {result_number}')
    plt.axis('off')
    
    # 個別数字表示（最大6個）
    for i, digit in enumerate(digits[:6]):
        plt.subplot(2, 3, 3+i)
        plt.imshow(digit, cmap='gray')
        plt.title(f'{i+1}: {predictions[i]} ({confidences[i]:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 信頼度の低い予測を警告
    low_conf_threshold = 0.7
    low_conf_digits = [(i, pred, conf) for i, (pred, conf) in enumerate(zip(predictions, confidences)) if conf < low_conf_threshold]
    
    if low_conf_digits:
        print(f"\n⚠️  信頼度が低い予測 (< {low_conf_threshold}):")
        for pos, pred, conf in low_conf_digits:
            print(f"  位置 {pos+1}: {pred} (信頼度: {conf:.3f})")

if __name__ == "__main__":
    main()