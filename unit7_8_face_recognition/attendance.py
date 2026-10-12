"""
[미니 프로젝트 ★★★] 얼굴 인식 출석부
교안: 통합 교안 8.5절 (실습 5·6 응용)

▶ 준비 : 웹캠, img/known/ 의 사진 (스마트폰·태블릿에 유명인 사진을 띄워 웹캠에 비춰 보세요)
▶ 실행 : python attendance.py
▶ 결과 : 인식된 사람의 이름과 시간이 attendance.csv 에 한 번씩 기록됨 (q 키로 종료)

실습 6(lab6_webcam.py)에 [출석 기록] 부분만 더한 프로그램입니다.
"""
import csv
from datetime import datetime

import cv2
import numpy as np
import face_recognition

# ── [출석 기록] 이미 출석한 사람은 다시 적지 않기 ──
checked = set()

def mark_attendance(name):
    if name == 'Unknown' or name in checked:
        return
    checked.add(name)
    with open('attendance.csv', 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow([name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    print(f'{name} 출석 완료!')

# ── 1. 등록하기 ──
known_people = {
    'Bill Gates': './img/known/bill.jpg',
    'Steve Jobs': './img/known/steve.jpg',
    'Elon Musk':  './img/known/elon.jpg',
    'Messi':      './img/known/messi.jpg',
    # 'Me':       './img/known/me.jpg',    # 내 사진을 넣고 맨 앞 #을 지우면 나도 등록!
}
known_face_encodings, known_face_names = [], []
for name, path in known_people.items():
    img = face_recognition.load_image_file(path)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    known_face_names.append(name)

# ── 2. 웹캠 켜기 ──
video_capture = cv2.VideoCapture(0)

face_locations = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print('카메라를 열 수 없어요.')
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = 'Unknown'
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
            face_names.append(name)
            mark_attendance(name)                  # ← 실습 6에 추가된 한 줄

    process_this_frame = not process_this_frame

    # ── 3. 결과 그리기 ──
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
