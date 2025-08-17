"""
자모(자음+모음) 단일 손 랜드마크(42차원) 분류 모델 학습 스크립트.

목적 :
- 클래스별 디렉터리 구조로 저장된 .npy 샘플을 로드하여 전처리한 뒤,
  간단한 MLP 분류 모델을 학습/평가하고, 학습된 모델과 레이블 인코더를 저장한다.

입력 형식 :
- 각 샘플은 길이 42 벡터(21 랜드마크 x (x, y)).
- 두 손 입력(84차원)으로 확장하려면 전처리/모델/데이터 파이프라인의 규격을 모두 변경해야 한다.

출력 파일:
- sign_model_combined.h5: 학습 완료된 Keras 모델
- label_encoder_combined.pkl: 클래스명 ↔ 정수 라벨 매핑 객체
"""
 
import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from utils import preprocess  # 전처리 함수(위치/스케일 정규화)

# 1) 데이터 경로 설정 : 자음+모음 샘플이 클래스별 폴더로 정리된 최상위 폴더
data_root = "combined_dataset"

# 2) 데이터 로딩 : 하위 폴더까지 재귀 탐색, .npy 파일만 사용, 전처리 포함
X, y = [], []
for root, dirs, files in os.walk(data_root):
    for fname in files:
        if fname.endswith(".npy"):
            path = os.path.join(root, fname)
            label = os.path.basename(os.path.dirname(path))  # 상위 폴더명이 곧 클래스명
            data = np.load(path)
            if data.shape == (42,):
                data = preprocess(data)  # 전처리 적용(학습/추론 파이프라인 정합성 유지)
                X.append(data)
                y.append(label)

X = np.array(X)
y = np.array(y)

# 3) 레이블 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 4) 레이블 인코더 저장 : 추론 시 인덱스 → 클래스명 복원에 필요
with open("label_encoder_combined.pkl", "wb") as f:
    pickle.dump(le, f)

# 5) 데이터 분할 : 클래스 분포 보존을 위해 stratify에 정수 라벨 사용
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6) 모델 정의 : 42차원 입력 MLP, 과적합 방지를 위한 Dropout 포함
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7) EarlyStopping 콜백 : 검증 손실 기준, 베스트 가중치 복원
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 8) 모델 학습
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# 9) 평가 및 모델 저장
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy : {acc:.2%}")

model.save("sign_model_combined.h5")
print("Saved model to sign_model_combined.h5")
