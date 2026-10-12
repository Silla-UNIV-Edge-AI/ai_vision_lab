"""
[실습 0-B] 얼굴 안에서 눈까지 찾기 – Haar Cascade + ROI
교안: 통합 교안 7.3.5

▶ 준비 : img/haar/group.jpg  (여러 사람이 나온 사진 – 오바마·바이든)
▶ 실행 : python lab0b_haar_face_eye.py
▶ 결과 : 얼굴은 파란 네모, 눈은 초록 네모 (아무 키나 누르면 종료)
"""
import cv2

# 1) 얼굴용·눈용 분류기 불러오기
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)
eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_xml)

# 2) 사진 불러와서 흑백으로 바꾸기 (Haar는 밝기 차이만 봄)
img = cv2.imread('./img/haar/group.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) 사진에서 얼굴 찾기 (scaleFactor=1.3, minNeighbors=5)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # 4) 얼굴에 파란 네모 그리기
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 5) 얼굴 부분만 잘라 낸 ROI(관심 영역) 만들기
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # 6) ROI 안에서만 눈 찾기 → 눈에 초록 네모 (ROI에 그리면 원본에도 반영됨)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# 7) 결과 보여 주기 (아무 키나 누르면 종료)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
