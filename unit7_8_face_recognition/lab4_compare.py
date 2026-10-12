"""
[실습 4] 얼굴 비교하기 – 같은 사람일까?
교안: 통합 교안 8.1절

▶ 준비 : img/known/messi.jpg (기준 사진), img/unknown/messi2.jpg (같은 사람일까?)
▶ 실행 : python lab4_compare.py
▶ 결과 : 얼굴 거리와 '같은 사람/다른 사람' 판정이 출력됨
"""
import face_recognition

# 1) 기준이 되는 사람(아는 사람)의 얼굴 바코드 만들기
known_image = face_recognition.load_image_file('./img/known/messi.jpg')
known_encoding = face_recognition.face_encodings(known_image)[0]

# 2) 맞혀 볼 사진의 얼굴 바코드 만들기
unknown_image = face_recognition.load_image_file('./img/unknown/messi2.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# 3) 비교하기 (tolerance: 합격 기준선, 기본값 0.6)
results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)
distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

# 4) 결과 출력
print(f'얼굴 거리: {distance:.3f}')
if results[0]:
    print('같은 사람입니다!')
else:
    print('다른 사람입니다.')
