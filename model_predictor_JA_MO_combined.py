"""
실시간 한글 자모(자음+모음) 수어 예측기

목적 :
- 웹캠 프레임을 캡처하고 MediaPipe Hands로 손 랜드마크를 추출한 뒤,
  학습 파이프라인과 동일하게 전처리하여 Keras 모델로 단일 자모를 예측한다.
  예측 결과는 로마자 표기와 신뢰도로 화면에 표시한다.

준비 파일 :
- sign_model_combined.h5: JA+MO 분류용 학습된 Keras 모델
- label_encoder_combined.pkl: 클래스 인덱스 ↔ 한글 자모 매핑을 담은 LabelEncoder
- utils.preprocess(list[float]) -> np.ndarray: 길이 42(랜드마크 21개 × 좌표 2)의 벡터를 모델 입력 형태/범위로 변환한다.

조작 :
- 'q' 키를 눌러 종료.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from utils import preprocess 

# 자모 → 로마자 표기 매핑 (UI 표시용). 예측된 자모를 사람이 읽기 쉽게 표시.
label_map = {
    "ㄱ": "Giyeok", "ㄴ": "Nieun", "ㄷ": "Digeut", "ㄹ": "Rieul", "ㅁ": "Mieum",
    "ㅂ": "Bieup", "ㅅ": "Siot", "ㅇ": "Ieung", "ㅈ": "Jieut", "ㅊ": "Chieut",
    "ㅋ": "Kieuk", "ㅌ": "Tieut", "ㅍ": "Pieup", "ㅎ": "Hieut",
    "ㅏ": "Ah", "ㅑ": "Ya", "ㅓ": "Eo", "ㅕ": "Yeo", "ㅗ": "O", "ㅛ": "Yo",
    "ㅜ": "U", "ㅠ": "Yu", "ㅡ": "Eu", "ㅣ": "I",
    "ㅐ": "Ae", "ㅔ": "E", "ㅚ": "Oe", "ㅟ": "Wi", "ㅒ": "Yae", "ㅖ": "Ye", "ㅢ": "Ui"
}

# 1) 모델 및 라벨 인코더 로드 : 현재 작업 디렉터리 기준 경로 사용
model = load_model("sign_model_combined.h5")
with open("label_encoder_combined.pkl", "rb") as f:
    le = pickle.load(f)

# MediaPipe Hands 초기화 : 기본 파라미터 사용(필요 시 성능/정확도 조정)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
 
# 2) 웹캠(기본 장치) 열기
cap = cv2.VideoCapture(0)

print("Real-time Sign Prediction Started (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

# 3) 정사각 크롭 : 종횡비 차이에 따른 좌표 스케일 편차를 줄여 일관성 확보
    h, w, _ = frame.shape
    min_dim = min(h, w)
    frame = frame[0:min_dim, 0:min_dim]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    frame_data = []
    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            # 4) 21개 랜드마크의 (x, y) 좌표를 순서대로 수집 → 총 42개 값 : 지화이기 때문에 단일 손 가정
            for lm in hand.landmark:
                frame_data.extend([lm.x, lm.y])

        # 5) 가드 : 전처리/모델 입력은 42길이(21×2)를 기대
        if len(frame_data) == 42:
            data = preprocess(frame_data)
            pred = model.predict(np.expand_dims(data, axis=0), verbose=0)
            class_index = np.argmax(pred)
            confidence = pred[0][class_index] * 100
            predicted_label = le.inverse_transform([class_index])[0]
            eng_label = label_map.get(predicted_label, predicted_label)
            prediction_text = f"{eng_label} ({confidence:.1f}%)"

    # 6) 예측 결과 오버레이(UI)
    cv2.putText(frame, f"Prediction : {prediction_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Sign Predictor (Romanized)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()
hands.close()
