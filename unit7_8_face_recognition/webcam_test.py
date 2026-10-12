"""
[환경 점검] 웹캠 미리 테스트하기
교안: 통합 교안 7.2.7

▶ 실행 : python webcam_test.py
▶ 결과 : 웹캠 화면이 뜨면 성공 (q 키를 누르면 종료)
"""
import cv2

cap = cv2.VideoCapture(0)          # 안 되면 1, 2로 바꿔 보기
while True:
    ok, frame = cap.read()
    if not ok:
        print('카메라를 열 수 없어요.')
        break
    cv2.imshow('Webcam Test - q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
