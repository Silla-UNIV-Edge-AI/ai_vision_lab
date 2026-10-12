"""
[실습 6] 실시간 얼굴 인식 – 웹캠으로 나를 알아보는 AI
교안: 통합 교안 8.3절 (Jetson Orin Nano에서도 그대로 실행: 8.4절)

▶ 준비 : 웹캠, img/known/ 의 사진 (스마트폰·태블릿에 유명인 사진을 띄워 웹캠에 비춰 보세요)
▶ 실행 : python lab6_webcam.py
▶ 결과 : 웹캠 화면의 얼굴에 빨간 네모와 이름 (q 키를 누르면 종료)
"""
import cv2
import numpy as np
import face_recognition

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

# ── 2. 웹캠 켜기 (0번 = 기본 카메라, 안 되면 1로 바꾸기) ──
video_capture = cv2.VideoCapture(0)

face_locations = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()          # 한 장면 가져오기
    if not ret:
        print('카메라를 열 수 없어요.')
        break

    if process_this_frame:
        # 1/4 크기로 줄이고 BGR → RGB 변환
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = 'Unknown'
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)   # 가장 닮은 사람 번호
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame   # 한 장씩 건너뛰기

    # ── 3. 결과 그리기 ──
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4      # 1/4로 줄였던 좌표를 원래 크기로
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      # q 키를 누르면 종료
        break

video_capture.release()
cv2.destroyAllWindows()
