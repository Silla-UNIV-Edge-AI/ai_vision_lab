"""
[환경 점검] 설치가 제대로 되었는지 한 번에 확인하기
교안: 통합 교안 7.2.7

▶ 실행 : python check_env.py
▶ 결과 : 버전이 모두 출력되고 마지막에 '모든 준비 완료!'가 나오면 성공
"""
import sys
print('Python          :', sys.version.split()[0])

import cv2
print('OpenCV          :', cv2.__version__)

import dlib
print('dlib            :', dlib.__version__, '| GPU(CUDA) 사용:', dlib.DLIB_USE_CUDA)

import face_recognition
print('face_recognition:', face_recognition.__version__)

import numpy, PIL
print('NumPy           :', numpy.__version__)
print('Pillow          :', PIL.__version__)

# 웹캠 연결 확인
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
print('웹캠            :', '정상' if ok else '연결 안 됨 (실습 0-C·6 전에 확인 필요)')

print('\n모든 준비 완료!')
