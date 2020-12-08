from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound
from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time

# facenet : 얼굴을 찾는 모델
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model : 마스크 검출 모델
model = load_model('models/mask_detector.model')

status_mask = False
status_qr = False

# 동영상 파일 읽기
# cap = cv2.VideoCapture('imgs/01.mp4')
# 실시간 웹캠 읽기
cap = cv2.VideoCapture(0)  # 웹캠 읽기만함 영상 띄워주는게 아님
i = 0

while cap.isOpened():
    ret, img = cap.read()   # img -> 실시간 영상을 이미지로 저장
    if status_mask == False:
        if not ret:
            break
        # 이미지의 높이와 너비 추출
        h, w = img.shape[:2]

        # 이미지 전처리
        # ref. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

        # facenet의 input으로 blob을 설정
        facenet.setInput(blob)
        # facenet 결과 추론, 얼굴 추출 결과가 dets의 저장
        dets = facenet.forward()

        # 한 프레임 내의 여러 얼굴들을 받음
        result_img = img.copy()     # 얼굴 영역

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
            print("mask : {0} | nomask : {1}".format(mask, nomask))

            # 마스크를 꼈는지 안겼는지에 따라 라벨링해줌
            if (mask - nomask) > 0.3:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
                print("마스크 검출")
                status_mask = True
                cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
                cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow('img', result_img)
                break
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
                print("마스크 없음")
                # frequency = 2500  # Set Frequency To 2500 Hertz
                # duration = 1000  # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
                cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
                cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow('img', result_img)

        cv2.putText(result_img, text="Show me th Mask", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 255, 0), thickness=2)
        # if status_mask == True:
        #     cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        #     cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
        #                 color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('img', result_img)

        if status_mask == True:
            time.sleep(1.0)
            cv2.destroyWindow('img')
    elif (status_mask == True) and (status_qr == False):
        # find the barcodes in the frame and decode each of the barcodes
        # 프레임에서 바코드를 찾고, 각 바코드들 마다 디코드
        barcodes = pyzbar.decode(img)

        ### Let’s proceed to loop over the detected barcodes
        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw
            # the bounding box surrounding the barcode on the image
            # 이미지에서 바코드의 경계 상자부분을 그리고, 바코드의 경계 상자부분(?)을 추출한다.
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # the barcode data is a bytes object so if we want to draw it
            # on our output image we need to convert it to a string first
            # 바코드 데이터는 바이트 객체이므로, 어떤 출력 이미지에 그리려면 가장 먼저 문자열로 변환해야 한다.
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # draw the barcode data and barcode type on the image
            # 이미지에서 바코드 데이터와 테입(유형)을 그린다
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(barcodeData)
            status_qr = True
            time.sleep(1.0)
            cv2.destroyWindow('QRScanner')
            time.sleep(1.0)
            break

            # if the barcode text is currently not in our CSV file, write
            # the timestamp + barcode to disk and update the set
            # 현재 바코드 텍스트가 CSV 파일안에 없을경우, timestamp, barcode를 작성하고 업데이트
            # if barcodeData not in found:
            #     csv.write("{},{}\n".format(datetime.datetime.now(),
            #                                barcodeData))
            #     csv.flush()
            #     found.add(barcodeData)

        if (status_mask == True) and (status_qr == True):
            status_mask = False
            status_qr = False
            continue
        # show the output frame
        cv2.putText(img, text="Show me th QRcode", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 255, 0), thickness=2)
        cv2.imshow("QRScanner", img)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break