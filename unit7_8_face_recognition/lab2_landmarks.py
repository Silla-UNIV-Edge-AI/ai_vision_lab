"""
[실습 2] 얼굴 랜드마크 – 눈·코·입에 점 찍기
교안: 통합 교안 7.5절

▶ 준비 : img/known/messi.jpg  (정면 얼굴 사진 – 메시)
▶ 실행 : python lab2_landmarks.py
▶ 결과 : 얼굴에 빨간 네모, 68개 특징점에 초록 점 (아무 키나 누르면 종료)
"""
import cv2
import face_recognition

# 1) 사진 불러오기 (face_recognition은 RGB 순서로 읽음)
image = face_recognition.load_image_file('./img/known/messi.jpg')

# 2) 얼굴 위치와 랜드마크 찾기
face_locations = face_recognition.face_locations(image)
landmarks_list = face_recognition.face_landmarks(image)
print(landmarks_list)

# 3) OpenCV는 BGR 순서를 쓰므로 색 순서 바꾸기
canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 4) 얼굴에 빨간 네모 그리기
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 255), 2)

# 5) 모든 부위의 점마다 초록 동그라미 그리기
for landmarks in landmarks_list:                 # 얼굴 하나씩
    for part_name, points in landmarks.items():  # 부위 하나씩 (chin, nose_tip ...)
        for point in points:                     # 점 하나씩
            cv2.circle(canvas, point, 3, (0, 255, 0), -1)

# 6) 결과 보여 주기 (아무 키나 누르면 종료)
cv2.imshow('result', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
