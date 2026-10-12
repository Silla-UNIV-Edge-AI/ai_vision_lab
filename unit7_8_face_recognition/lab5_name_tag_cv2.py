"""
[실습 5] 얼굴 인식하기 – 사진 속 사람에게 이름표 붙이기 (OpenCV 버전)
교안: 통합 교안 8.2.2

▶ 준비 : img/known/ 의 사진 6장, img/groups/bill-steve-elon.jpg
▶ 실행 : python lab5_name_tag_cv2.py
▶ 결과 : 얼굴마다 초록 네모와 영어 이름표 (아무 키나 누르면 종료)
"""
import cv2
import face_recognition

# ── 1. 등록하기: 이름과 사진 경로 ──
known_people = {
    'Bill Gates':  './img/known/bill.jpg',
    'Steve Jobs':  './img/known/steve.jpg',
    'Elon Musk':   './img/known/elon.jpg',
    'Messi':       './img/known/messi.jpg',
    'Obama':       './img/known/obama.jpg',
    'Biden':       './img/known/biden.jpg',
}

known_face_encodings = []
known_face_names = []
for name, path in known_people.items():
    img = face_recognition.load_image_file(path)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    known_face_names.append(name)

# ── 2. 찾기: 단체 사진의 얼굴 위치와 바코드 ──
test_image = face_recognition.load_image_file('./img/groups/bill-steve-elon.jpg')
test_locations = face_recognition.face_locations(test_image)
test_encodings = face_recognition.face_encodings(test_image, test_locations)

canvas = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)   # OpenCV용 BGR로 변환

# ── 3. 대조하기 + 4. 표시하기 ──
for (top, right, bottom, left), encoding in zip(test_locations, test_encodings):
    name = 'Unknown'
    matches = face_recognition.compare_faces(known_face_encodings, encoding, 0.6)
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(canvas, (left, top - 25), (right, top), (0, 255, 0), cv2.FILLED)
    cv2.putText(canvas, name, (left + 4, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow('result', canvas)
cv2.waitKey(0)            # 아무 키나 누르면 종료
cv2.destroyAllWindows()
