"""
[3.4 ④] 구간별 시간 측정 템플릿 (병목 찾기)
교안: 3단원 3.4

▶ 실행 : 내 실시간 앱의 해당 위치에 복사해서 사용
"""
# ※ preprocess / infer / postprocess 는 내 앱의 함수 이름으로 바꿔서 쓰는 "틀"입니다. 그대로 실행하는 파일이 아닙니다.
import time

import torch

t0 = time.perf_counter()
blob = preprocess(frame)          # 전처리
t1 = time.perf_counter()
out = infer(blob)                 # 모델 추론
torch.cuda.synchronize()          # GPU 는 비동기로 돌아서, 계산이 끝날 때까지 기다린 뒤 시간을 재야 정확함
t2 = time.perf_counter()
boxes = postprocess(out)          # 후처리
t3 = time.perf_counter()

print(f'pre {(t1-t0)*1000:.1f} ms | infer {(t2-t1)*1000:.1f} ms | post {(t3-t2)*1000:.1f} ms')


# ── ultralytics YOLO 를 쓴다면: 구간별 시간(ms)이 결과에 이미 들어 있습니다 ──
# from ultralytics import YOLO
# model = YOLO('best.engine')
# for r in model(source=0, imgsz=416, stream=True):
#     print(r.speed)              # {'preprocess': 2.1, 'inference': 9.6, 'postprocess': 1.3}
