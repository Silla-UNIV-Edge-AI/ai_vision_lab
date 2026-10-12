"""
[미니 프로젝트 ★] 스마일 셀카
교안: 통합 교안 8.5절 (실습 0 응용)

▶ 준비 : 웹캠
▶ 실행 : python smile_selfie.py
▶ 결과 : 웃으면 'Smile!' 표시와 함께 selfies 폴더에 사진이 저장됨 (q 키로 종료)

실습 0-C(웹캠 얼굴 찾기) + 실습 0-B(얼굴 ROI 안에서 찾기)를 합친 프로그램입니다.
"""
import os
import time
import cv2

# 1) 얼굴용·웃음용 분류기 불러오기
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)
smile_xml = cv2.data.haarcascades + 'haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(smile_xml)

os.makedirs('selfies', exist_ok=True)
last_saved = 0                       # 마지막으로 저장한 시각 (연속 저장 방지)

# 2) 웹캠 켜기
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print('카메라를 열 수 없어요.')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) 얼굴 찾기
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 4) 얼굴 아래쪽 절반(ROI)에서만 웃음 찾기
        roi_gray = gray[y + h // 2:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)

        # 5) 웃음을 찾으면 표시하고, 2초에 한 번만 사진 저장
        if len(smiles) > 0:
            cv2.putText(img, 'Smile!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if time.time() - last_saved > 2:
                filename = time.strftime('selfies/smile_%Y%m%d_%H%M%S.jpg')
                cv2.imwrite(filename, img)
                print(f'저장했어요: {filename}')
                last_saved = time.time()

    cv2.imshow('smile selfie', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
