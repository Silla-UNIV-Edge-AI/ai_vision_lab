"""
[실습 5 – Pillow 버전] 한글 이름표 붙이기
교안: 통합 교안 8.2.3

▶ 준비 : lab5_name_tag_cv2.py 와 같은 사진 + 한글 글꼴
         Windows 'malgun.ttf' / Mac '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
▶ 실행 : python lab5_name_tag_pil.py
▶ 결과 : 얼굴마다 노란 네모와 한글 이름표가 그려진 사진 창
"""
import face_recognition
from PIL import Image, ImageDraw, ImageFont

# ── 1. 등록하기 (한글 이름도 가능) ──
known_people = {
    '빌 게이츠':   './img/known/bill.jpg',
    '스티브 잡스': './img/known/steve.jpg',
    '일론 머스크': './img/known/elon.jpg',
    '메시':        './img/known/messi.jpg',
    '오바마':      './img/known/obama.jpg',
    '바이든':      './img/known/biden.jpg',
}

known_face_encodings = []
known_face_names = []
for name, path in known_people.items():
    img = face_recognition.load_image_file(path)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    known_face_names.append(name)

# ── 2. 찾기 ──
test_image = face_recognition.load_image_file('./img/groups/bill-steve-elon.jpg')
test_locations = face_recognition.face_locations(test_image)
test_encodings = face_recognition.face_encodings(test_image, test_locations)

# ── 3. 대조하기 + 4. 표시하기 (Pillow로 그리기) ──
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)
font = ImageFont.truetype('malgun.ttf', 20)   # Windows 맑은 고딕 (Mac은 위 경로 사용)

for (top, right, bottom, left), encoding in zip(test_locations, test_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, encoding)
    name = '모르는 사람'
    if True in matches:
        name = known_face_names[matches.index(True)]

    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=3)
    # 글자 크기 구하기 (Pillow 10 이상은 textsize 대신 textbbox 사용)
    x0, y0, x1, y1 = draw.textbbox((0, 0), name, font=font)
    text_height = y1 - y0
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)),
                   fill=(255, 255, 0), outline=(255, 255, 0))
    draw.text((left + 6, bottom - text_height - 8), name, fill=(0, 0, 0), font=font)

del draw
pil_image.show()
