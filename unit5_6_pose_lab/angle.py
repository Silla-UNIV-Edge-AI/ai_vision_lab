"""
[실습 4] 세 점으로 각도 구하기 (실습 5 · 6에서 재사용)
교안: 6단원 6.1

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2
▶ 실행 : python3 angle.py
▶ 결과 : 90.0 / 180.0 / 약 5.7
"""
# 실습 4. 세 점으로 각도 구하기
import math

def get_angle(a, b, c):
    """b를 꼭짓점으로 하는 각도 (a-b-c), 0~180도"""
    dir1 = math.atan2(a[1] - b[1], a[0] - b[0])   # b에서 a를 바라보는 방향
    dir2 = math.atan2(c[1] - b[1], c[0] - b[0])   # b에서 c를 바라보는 방향
    angle = abs(math.degrees(dir1 - dir2))        # 두 방향의 차이
    if angle > 180:
        angle = 360 - angle                       # 항상 작은 쪽 각도로
    return angle

if __name__ == "__main__":                        # 이 파일을 직접 실행할 때만 테스트
    print(get_angle((0, 0), (10, 0), (10, 10)))   # ㄱ자  -> 90.0
    print(get_angle((0, 0), (10, 0), (20, 0)))    # 일자  -> 180.0
    print(get_angle((0, 0), (10, 0), (0, 1)))     # 거의 접힘 -> 약 5.7
