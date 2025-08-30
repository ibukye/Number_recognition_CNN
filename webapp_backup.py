# FastAPI
from fastapi import FastAPI
#
from pydantic import BaseModel
#
import base64, io
#
from PIL import Image
#
import numpy as np
# 
from tensorflow.keras.models import load_model

from MDP_function import MDP

#
from fastapi.responses import HTMLResponse

import cv2

#
app = FastAPI()

class ImageData(BaseModel):
    image: str

@app.get("/")
def index():
    with open("try.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


#
@app.post("/predict")
def predict(data: ImageData):
    try:
        print("Starting prediction process...")
        
        # "data:image/png;base64,..."からBase64部分だけを取り出す
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img_array = np.array(img, dtype=np.uint8)
        
        print("Original img_array shape:", img_array.shape)
        print("Original img_array mean:", np.mean(img_array))
        
        # HTML側で黒い線で描画されているので、反転処理を削除
        # MNISTは白い数字、黒い背景が期待されるので、画像を反転
        img_array = 255 - img_array
        
        # デバッグ用：処理後の画像を保存
        cv2.imwrite("processed_image.png", img_array)
        
        print("After inversion - img_array mean:", np.mean(img_array))
        print("Calling MDP function...")
        
        if MDP is None:
            return {"error": "MDP function not available"}

        result = MDP(img_array)
        print("predicted number:", result)
        return {"result": result}
        
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}
