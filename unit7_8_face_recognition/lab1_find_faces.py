"""
[실습 1] 얼굴 찾기 – 사진 속에 몇 명이 있을까?
교안: 통합 교안 7.4절

▶ 준비 : img/groups/team2.jpg  (단체 사진 – 11명)
▶ 실행 : python lab1_find_faces.py
▶ 결과 : 얼굴 좌표 목록과 사람 수가 출력됨
"""
import face_recognition

# 1) 단체 사진 불러오기
image = face_recognition.load_image_file('./img/groups/team2.jpg')

# 2) 사진 속 모든 얼굴의 위치 찾기 → [(top, right, bottom, left), ...]
face_locations = face_recognition.face_locations(image)

# 3) 결과 출력
print(face_locations)
print(f'이 사진에는 {len(face_locations)}명이 있어요!')
