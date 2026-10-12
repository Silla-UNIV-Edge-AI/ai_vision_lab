"""
[실습 4 – OpenCV 버전] OpenCV로 사진을 읽어 비교하기
교안: 통합 교안 8.1.2

▶ 준비 : img/known/messi.jpg, img/unknown/messi2.jpg
▶ 실행 : python lab4_compare_cv2.py
▶ 결과 : Result: True(같은 사람) 또는 False 출력 + 두 사진 창 (아무 키나 누르면 종료)
"""
import cv2
import face_recognition

# 1) OpenCV로 읽기 → BGR을 RGB로 바꾼 뒤 얼굴 바코드 만들기
img = cv2.imread('./img/known/messi.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread('./img/unknown/messi2.jpg')
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# 2) 비교하기
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print('Result:', result[0])

# 3) 두 사진 보여 주기 (아무 키나 누르면 종료)
cv2.imshow('Img 1', img)
cv2.imshow('Img 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
