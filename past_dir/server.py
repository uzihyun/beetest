from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# 모델 로드 (미리 학습된 .keras 모델 파일 경로 설정)
MODEL_PATH = './accuracy_85.keras'  # 여기에 실제 .keras 파일 경로를 입력하세요.
model = load_model(MODEL_PATH)

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

        # 이미지를 모델이 인식할 수 있는 형태로 변환
        img = image.load_img(filepath, target_size=(244, 244))  # 모델 입력 크기에 맞게 조정
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        predictions = model.predict(img_array)

        # 클래스 리스트
        classlist = ["cans", "glass", "other_ps", "p_bowls", "pets"]

        # 가장 높은 확률을 가진 클래스의 인덱스 추출
        result_index = np.argmax(predictions, axis=1)[0]
        answer = classlist[result_index]  # 인덱스에 해당하는 클래스를 설정

        # 결과와 함께 이미지 경로 전달
        return render_template('result.html', prediction=answer, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
