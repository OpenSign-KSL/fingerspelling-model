"""
랜드마크 전처리 유틸리티

목적 :
- MediaPipe 등에서 얻은 손 랜드마크 (x, y) 좌표 시퀀스를 학습 파이프라인과 동일하게
  정규화한다. 위치 불변성(translation invariance)과 크기 불변성(scale invariance)을 부여한다.

입력 :
- `landmarks`: 길이 2×N의 1차원 배열/리스트. (x1, y1, x2, y2, ...)
- 단일 손(21포인트)일 경우 길이 42, 두 손(42포인트)일 경우 길이 84도 처리 가능.
  단, 현재 구현은 전체 시퀀스를 하나의 집합으로 정규화하므로, 두 손을 각각 별도로 정규화하려면 손별로 분리하여 함수들을 적용해야 한다.

출력 :
- 동일 길이의 1차원 `np.ndarray[float]` 반환. (N, 2) → 평탄화.
"""

import numpy as np

def normalize_landmarks(landmarks):
    """
    기준점(첫 좌표)을 원점(0, 0)으로 평행이동하여 위치 영향 제거.

    주의점 :
    - MediaPipe Hands 기준, 첫 랜드마크는 보통 손목(wrist)이다.
    - 두 손을 함께 입력한 경우 첫 포인트(예: 왼손 손목)만 기준이 되므로, 손별 독립 정규화를 원하면 손별로 분리하여 호출해야 함.
    """
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]  # 손목 기준
    normalized = landmarks - base
    return normalized.flatten()

def scale_normalize_landmarks(landmarks):
    """
    좌표를 중심화(평균 0)한 뒤, 최대 반지름으로 나누어 스케일을 정규화.

    - 중심화 : 다양한 포인트 분포에서 공통 기준을 맞춤
    - 스케일 정규화 : 최대 거리(유클리드 노름)로 나눠 크기 차이를 완화
    - 1e-6 : 영(0) 나눗셈 방지용 작은 안정화 항
    """
    landmarks = np.array(landmarks).reshape(-1, 2)
    center = np.mean(landmarks, axis=0)
    landmarks -= center
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    normalized = landmarks / (max_dist + 1e-6)
    return normalized.flatten()

def preprocess(landmarks):
    """
    학습 파이프라인과 동일한 전처리 파이프라인.

    단계 :
    1) 기준점 기준 평행이동(normalize_landmarks)
    2) 중심화 및 스케일 정규화(scale_normalize_landmarks)

    반환 :
    - 정규화된 (N, 2) 평탄화 벡터
    """
    # 방법 1 + 2 조합
    norm = normalize_landmarks(landmarks)
    return scale_normalize_landmarks(norm)
