"""
[실습 3] 얼굴 가져오기 – 얼굴만 오려서 저장하기
교안: 통합 교안 7.6절

▶ 준비 : img/groups/team1.jpg  (단체 사진 – 4명)
▶ 실행 : python lab3_crop_faces.py
▶ 결과 : faces 폴더에 face_0.jpg ~ face_3.jpg 가 저장됨
"""
import os
from PIL import Image
import face_recognition

# 1) 저장할 폴더 만들기
os.makedirs('faces', exist_ok=True)

# 2) 단체 사진에서 얼굴 위치 찾기
image = face_recognition.load_image_file('./img/groups/team1.jpg')
face_locations = face_recognition.face_locations(image)

# 3) 얼굴마다 잘라서 보여 주고 저장하기
for i, (top, right, bottom, left) in enumerate(face_locations):
    face_image = image[top:bottom, left:right]   # [세로 범위, 가로 범위]로 잘라 내기

    pil_image = Image.fromarray(face_image)      # 배열 → 이미지로 변환
    pil_image.show()                             # 화면에 보여 주기
    pil_image.save(f'faces/face_{i}.jpg')        # 파일로 저장

print(f'{len(face_locations)}개의 얼굴을 faces 폴더에 저장했어요.')
