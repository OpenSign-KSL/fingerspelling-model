import cv2
import mediapipe as mp
import numpy as np
import os
import re
from utils import preprocess

# 한글 모음 리스트
# 한글 모음 + 대응 영문표기
vowels = list("ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅔㅚㅟㅒㅖㅢ")
vowel_labels = [
    "A", "YA", "EO", "YEO", "O", "YO", "U", "YU", "EU", "I",
    "AE", "E", "OE", "WI", "YAE", "YE", "UI"
]

# 초기 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

current_index = 0
save_root = "dataset"

def get_next_sample_index(folder):
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    indices = [int(re.findall(r'\d+', f)[0]) for f in files if re.findall(r'\d+', f)]
    return max(indices) + 1 if indices else 1

print("✅ 모음 수집기 시작!")
print("▶ 's': 저장 | 'n': 다음 모음 | 'q': 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    min_dim = min(h, w)
    frame = frame[0:min_dim, 0:min_dim]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    frame_data = []
    detected = False

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            detected = True
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            for lm in hand.landmark:
                frame_data.extend([lm.x, lm.y])

    # 현재 모음 표시 (한글)
    current_vowel = vowels[current_index]
    folder = os.path.join(save_root, current_vowel)
    next_index = get_next_sample_index(folder)
    display_text = f"[{vowel_labels[current_index]}] sample_{next_index:03d}.npy | s:save n:next q:quit"

    cv2.putText(frame, display_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Vowel Collector - Korean", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and detected and len(frame_data) == 42:
        data = preprocess(frame_data)
        filename = os.path.join(folder, f"sample_{next_index:03d}.npy")
        np.save(filename, data)
        print(f"✅ 저장됨: {filename}")

    elif key == ord('n'):
        current_index += 1
        if current_index >= len(vowels):
            print("🎉 모든 모음 수집 완료!")
            break
        print(f"▶ 다음 모음으로 이동: {vowels[current_index]}")

    elif key == ord('q'):
        print("🛑 수집 종료")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
