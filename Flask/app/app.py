from flask import Flask, request, jsonify
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model_path = "./model.pt"
model = YOLO(model_path)
threshold = 0.5

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        json_data = request.get_json()  # 클라이언트로부터 JSON 데이터를 받아옴
        if json_data is None or 'img' not in json_data:
            return jsonify({"error": "No image provided"}), 400  # 에러 메시지 반환

        img_data = json_data['img']
        img_data = base64.b64decode(img_data)  # base64 인코딩 해제
        img_data = BytesIO(img_data)  # 이미지 데이터를 바이트 스트림으로 변환
        img = Image.open(img_data)  # 이미지 객체 생성

        #예측
        results = model(img)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
                cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6, cv2.LINE_AA)

        # 이미지를 다시 base64 인코딩하여 전송
        buffered = BytesIO()
        img.save(buffered, format="JPEG")  # 이미지를 JPEG 형식으로 버퍼에 저장
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')  # 버퍼의 내용을 base64로 인코딩

        return jsonify({"width": img.size[0], "height": img.size[1], "encoded_image": img_base64})
    else:
        return jsonify({"message": "Send a POST request with image data"})
    

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)  # 서버 설정