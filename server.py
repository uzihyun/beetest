from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import torch
import clip
from PIL import Image


app = Flask(__name__)

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Logistic Regression 모델 로드
MODEL_PATH = './model.pkl'  # 저장된 Logistic Regression 모델 경로
classifier = joblib.load(MODEL_PATH)

# 업로드된 이미지를 저장할 경로
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 분석
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return '파일을 찾을 수 없습니다.'

    file = request.files['file']

    if file.filename == '':
        return '파일이 선택되지 않았습니다.'

    if file:
        # 파일을 저장
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 이미지를 CLIP 모델에 맞게 전처리
        img = Image.open(filepath).convert("RGB")
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)

        # CLIP 모델을 통해 특징 벡터 추출
        with torch.no_grad():
            image_features = model.encode_image(img_preprocessed).cpu().numpy()

        # Logistic Regression 모델로 예측
        prediction = classifier.predict(image_features)

        # 클래스 리스트 (학습 시 사용한 클래스 목록에 맞게 수정)
        classlist = ["binil", "cans", "glass", "other_ps", "pets"]

        # 예측 결과 가져오기
        result_index = prediction[0]
        answer = classlist[result_index]

        # 결과와 함께 이미지 경로 전달
        return render_template('result.html', prediction=answer, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
