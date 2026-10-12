"""
[실습 0-A] 사진 속 얼굴 찾기 – Haar Cascade
교안: 통합 교안 7.3.4

▶ 준비 : img/haar/single.jpg  (한 사람 사진 – 오바마 공식 사진)
▶ 실행 : python lab0a_haar_face.py
▶ 결과 : 얼굴에 자홍색 네모가 그려진 창 (아무 키나 누르면 종료)
"""
import cv2

# 1) 사진 불러오기
img = cv2.imread('./img/haar/single.jpg')

# 2) 분류기 만들기 + 학습된 XML 모델 불러오기
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)

# 3) 사진에서 얼굴 찾기 → [(x, y, w, h), ...]
faces = face_cascade.detectMultiScale(img)
print(f'찾은 얼굴 수: {len(faces)}')

# 4) 찾은 얼굴마다 네모 그리기 (왼쪽 위 → 오른쪽 아래)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

# 5) 결과 보여 주기 (아무 키나 누르면 종료)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
