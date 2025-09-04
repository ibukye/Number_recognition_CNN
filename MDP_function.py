from tensorflow.keras.models import load_model
import cv2
import numpy as np
import requests
import tempfile
import os

model = load_model("mnist_cnn_with_aug.keras")

MODEL_URL = "https://github.com/ibukye/Number_recognition_CNN/releases/download/v1.0.0/mnist_cnn_with_aug.keras"

def load_mnist_model():
    try:
        print("Downlooading model from GitHub in MDP function...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        # tmp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_model_path = temp_file.name

        model = load_model(temp_model_path)

        # delete tmp file
        os.unlink(temp_model_path)
        print("MNIST model loaded in MDP function")
    except Exception as e:
        print(f"Error loading model in MDP: {e}")
        model = None
    return model



def MDP(img_array):
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
        if (area < 100): continue
    
        without_rectangle = img_rectangled[y-10:y+h+10,x-10:x+w+10].copy()
        for_model.append(without_rectangle)
        cv2.rectangle(img_rectangled, (x,y), (x+w,y+h), (0,255,0), 2)
        digits.append(img_rectangled[y-10:y+h+10,x-10:x+w+10])

    """
    モデル予測
    """
    results = []
    i = 0
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
        
        pred = np.argmax(model.predict(input_img))
        results.append(str(pred))

        cv2.imwrite(f'detected_number_{i}.png', input_img)
        i+=1


    number = 0
    for i in range(len(results)):
        number += int(results[len(results) - (i+1)]) * (10 ** i)
    return number
    #print("認識結果: ", "".join(results))
