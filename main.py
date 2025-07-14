import cv2
import mediapipe as mp
import pyautogui

# Setup
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

click_down = False  # status klik

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Gambar titik
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Gerakkan kursor
            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Hitung jarak antara jempol dan telunjuk
            dist_x = index_tip.x - thumb_tip.x
            dist_y = index_tip.y - thumb_tip.y
            distance = (dist_x**2 + dist_y**2) ** 0.5

            # Klik jika cukup dekat
            if distance < 0.03:
                if not click_down:
                    click_down = True
                    pyautogui.click()
                    print("Klik!")
            else:
                click_down = False

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
