# OpenSign — 실시간 지화(자모) 예측 모델 소개

웹캠 영상에서 **MediaPipe Hands**로 손 랜드마크 **21점 × (x,y) = 42차원**을 추출해 전처리하고, **경량 MLP 분류기**로 **한글 자모(자음+모음)**를 **실시간** 예측합니다. 예측 결과는 로마자 표기와 신뢰도로 화면에 오버레이됩니다.

---

## 핵심 구성요소

- **학습 스크립트**: `JAMO Training.py`  
  - `.npy`(길이 42 벡터) 데이터 로드 → 전처리 → **MLP** 학습/검증 → 모델/라벨 저장
- **추론 스크립트**: `model_predictor_JA_MO_combined.py`  
  - 웹캠 → MediaPipe Hands → 42차원 벡터 → 전처리 → **Keras 모델 예측** → 화면 오버레이
- **모델 가중치**: `sign_model_combined.h5` (Keras)  
- **라벨 인코더**: `label_encoder_combined.pkl` (클래스 ↔ 자모 매핑)
- **전처리 유틸**: `utils.preprocess(list[float]) -> np.ndarray`

---

## 파이프라인(개요)

1. 입력 캡처: cv2.VideoCapture(0) 으로 웹캠 프레임 수집

2. 키포인트 추출: MediaPipe Hands로 21개 랜드마크 (x,y) 획득 → 길이 42의 벡터 구성

3. 전처리: utils.preprocess()로 학습과 동일한 스케일/정규화 적용 (학습·추론 정합성)

4. 분류: Keras MLP 모델(sign_model_combined.h5)이 자모 클래스를 예측

5. 라벨 디코딩/표시: label_encoder_combined.pkl로 클래스 ↔ 자모 매핑 복원, 로마자 표기 및 신뢰도 오버레이 출력

---

## 데이터셋 형식
- 최상위 폴더: combined_dataset/

- 클래스(자모)별 하위 폴더에 .npy 파일 저장
예) combined_dataset/ㄱ/sample_0001.npy, combined_dataset/ㅏ/sample_0020.npy …

- 각 .npy의 형태: (42,) ← 21 랜드마크 × (x,y)

- 스크립트가 재귀적으로 .npy를 찾아 로드하고, 상위 폴더명이 라벨이 됩니다.

---

## 학습(JAMO Training.py)
- 입력: combined_dataset/**/**.npy (재귀 탐색)

- 전처리: utils.preprocess로 학습/추론 정합성 확보

- 모델(예시): Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(#classes, softmax)

- 학습 설정: epochs=30, batch_size=32, EarlyStopping(patience=5, monitor=val_loss)

- 산출물

1. sign_model_combined.h5 (모델 가중치)

2. label_encoder_combined.pkl (라벨 인코더)

---

## 추론 스크립트 개요 (model_predictor_JA_MO_combined.py)

- 역할: 웹캠 프레임 → MediaPipe Hands → (x,y) 21점×2=42차원 벡터 생성 → 전처리 → Keras 모델 예측

- UI 출력: 로마자 표기 + 신뢰도(%)를 영상 위에 오버레이

- 필수 파일: sign_model_combined.h5, label_encoder_combined.pkl, utils.preprocess

- 종료 조작: q 키로 종료

- 참고: 프레임을 정사각형으로 크롭해 좌표 스케일 일관성을 유지합니다.

---

## 빠른 시작(실행 예시)

아래 명령은 Windows 기준 예시입니다. (macOS/Linux는 \ 대신 / 경로 구분)

1) 가상환경 & 의존성
```text
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

2) 학습(선택)
```text
# 폴더 예: combined_dataset/ㄱ/*.npy, combined_dataset/ㅏ/*.npy ...
python "JAMO Training.py"
# 완료 후: sign_model_combined.h5, label_encoder_combined.pkl 생성
```

3) 실시간 추론
```text
python model_predictor_JA_MO_combined.py
# 웹캠 허용 후, 화면에 "Prediction : {로마자} ({신뢰도}%)" 표시
# 'q' 로 종료
```

디렉터리 구조(예)
```text
OpenSign/
├─ combined_dataset/
│  ├─ ㄱ/  ├─ ㄴ/  ├─ ㅏ/  └─ ㅣ/ ...  # 각 폴더에 .npy(42차원) 파일
├─ sign_model_combined.h5              # 학습 결과 (추론 사용)
├─ label_encoder_combined.pkl          # 라벨 인코더 (추론 사용)
├─ JAMO Training.py                    # 학습 스크립트
├─ model_predictor_JA_MO_combined.py   # 실시간 추론 스크립트
└─ utils.py                            # preprocess() 등 전처리 유틸
```
