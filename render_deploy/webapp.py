# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
from PIL import Image
import numpy as np
from fastapi.responses import HTMLResponse
import cv2
from render_deploy.MDP_function_revised import MDP

# TensorFlowモデルのテスト読み込み
try:
    from tensorflow.keras.models import load_model
    model = load_model("mnist_cnn_with_aug.keras")
    print("Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    MODEL_LOADED = False

app = FastAPI()

class ImageData(BaseModel):
    image: str

@app.get("/")
def index():
    with open("try.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
def predict(data: ImageData):
    try:
        print("Starting prediction process...")
        
        # Base64デコード
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img_array = np.array(img, dtype=np.uint8)
        
        #print("Original img_array shape:", img_array.shape)
        #print("Original img_array mean:", np.mean(img_array))
        
        # 反転処理
        img_array = 255 - img_array
        #print("After inversion - img_array mean:", np.mean(img_array))
        
        # 画像保存（テスト用）
        #cv2.imwrite("processed_image.png", img_array)
        #print("Image saved successfully")
        
        # モデルが読み込まれているかテスト
        if not MODEL_LOADED:
            return {"result": "Model not loaded", "error": "TensorFlow model failed to load"}
        

        result = MDP(img_array)
        print("predicted number:", result)
        return {"result": result}

    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}