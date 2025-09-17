# ベースイメージ (軽量なPython環境)
FROM python:3.13-slim

# 作業Directory
WORKDIR /render_deploy

# 依存関係をinstall
COPY render_deploy/requirements.txt .

# OpenCVが必要とするライブラリを追加
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# アプリのsource codeとmodelをcopy
#COPY MDP_function_revised.py .
#COPY mnist_cnn_with_aug.keras .
#COPY webapp.py .
#COPY try.html .
COPY render_deploy .

# FastAPIを起動するcommand
CMD [ "uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000" ]