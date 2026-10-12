"""
[실습 0-C] 웹캠으로 실시간 얼굴 찾기 – Haar Cascade
교안: 통합 교안 7.3.6

▶ 준비 : 웹캠
▶ 실행 : python lab0c_haar_webcam.py
▶ 결과 : 웹캠 화면의 얼굴에 빨간 네모 (q 키를 누르면 종료)
"""
import cv2

# 1) 분류기 불러오기 (반복문 밖에서 한 번만!)
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)

# 2) 웹캠 켜기 (0번 = 기본 카메라, 안 되면 1로 바꾸기)
cap = cv2.VideoCapture(0)

while True:
    # 3) 한 장면 가져오기
    ret, img = cap.read()
    if not ret:
        print('카메라를 열 수 없어요.')
        break

    # 4) 크기 줄이고(속도↑) 흑백으로 바꾸기
    img = cv2.resize(img, (480, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 5) 얼굴 찾아 빨간 네모 그리기
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 6) 화면에 보여 주기 (q 키를 누르면 종료)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
