from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
import time
import gpio as GPIO
import mysql.connector
from playsound import playsound

config = {
    "user": "root",
    "password": "1234",
    "host": "127.0.0.1", #local
    "database": "mysql", #Database name
    "port": "3306" #port는 최초 설치 시 입력한 값(기본값은 3306)
}

# conn = mysql.connector.connect(**config)
# print(conn)
#     # db select, insert, update, delete 작업 객체
# cursor = conn.cursor()
#     # 실행할 select 문 구성
# sql = "SELECT * FROM users"
#     # cursor 객체를 이용해서 수행한다.
# cursor.execute(sql)
#     # select 된 결과 셋 얻어오기
# resultList = cursor.fetchall()  # tuple 이 들어있는 list
# print(resultList)

# facenet : 얼굴을 찾는 모델
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model : 마스크 검출 모델
model = load_model('models/mask_detector.model')

# 동영상 파일 읽기
# cap = cv2.VideoCapture('imgs/01.mp4')
# 실시간 웹캠 읽기
cap = cv2.VideoCapture(0)
i = 0

maskOrQr = False    # false일 경우 mask, true일 경우 qr
# run_servo = False
#
# def servoMotor(pin, degree, t):
#     GPIO.setmode(GPIO.BOARD)
#     GPIO.setup(pin, GPIO.OUT)
#     pwm = GPIO.PWM(pin, 50)
#
#     pwm.start(3)
#     time.sleep(t)
#
#     pwm.ChangeDutyCycle(degree)
#     time.sleep(t)
#     pwm.stop()
#     GPIO.cleanup(pin)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    if maskOrQr:
        cv2.putText(img, "Show me the QRCode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        decoded = pyzbar.decode(gray)

        for d in decoded:
            x, y, w, h = d.rect

            barcode_data = d.data.decode("utf-8")
            barcode_type = d.type

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            text = '%s (%s)' % (barcode_data, barcode_type)
            # if run_servo == False:
            #     servoMotor(16, 8, 1)
            #     run_servo = True
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            maskOrQr = False

        cv2.imshow('img', img)
        if maskOrQr == False:
            playsound('./audio/result1.mp3')
            time.sleep(0.5)

        else:
            time.sleep(0.1)

    else:
        run_servo = False
        # 이미지의 높이와 너비 추출
        h, w = img.shape[:2]

        # 이미지 전처리
        # ref. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

        # facenet의 input으로 blob을 설정
        facenet.setInput(blob)
        # facenet 결과 추론, 얼굴 추출 결과가 dets에 저장
        dets = facenet.forward()

        # 한 프레임 내의 여러 얼굴들을 받음
        result_img = img.copy()

        # 마스크를 찾용했는지 확인
        for i in range(dets.shape[2]):

            # 검출한 결과가 신뢰도
            confidence = dets[0, 0, i, 2]
            # 신뢰도를 0.5로 임계치 지정
            if confidence < 0.5:
                continue

            # 바운딩 박스를 구함
            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            # 원본 이미지에서 얼굴영역 추출
            face = img[y1:y2, x1:x2]

            # 추출한 얼굴영역을 전처리
            face_input = cv2.resize(face, dsize=(224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            # 마스크 검출 모델로 결과값 return
            mask, nomask = model.predict(face_input).squeeze()

            # 마스크를 꼈는지 안겼는지에 따라 라벨링해줌
            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
                maskOrQr = True
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second

            # 화면에 얼굴부분과 마스크 유무를 출력해해줌
            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(result_img, "Show me the Mask", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('img',result_img)
        if maskOrQr == True:
            time.sleep(0.5)
        else:
            time.sleep(0.1)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break