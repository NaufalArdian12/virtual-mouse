import cv2
import mediapipe as mp
import pyautogui

# Setup
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

click_down = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape

    status = "Tracking..."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Hitung jarak jempol dan telunjuk
            dist_x = index_tip.x - thumb_tip.x
            dist_y = index_tip.y - thumb_tip.y
            distance = (dist_x**2 + dist_y**2) ** 0.5

            if distance < 0.03:
                if not click_down:
                    click_down = True
                    pyautogui.click()
                    status = "Click!"
                color = (0, 0, 255)  # merah saat klik
            else:
                click_down = False
                color = (0, 255, 0)  # hijau saat normal

            # Gambar lingkaran di ujung jari
            cv2.circle(frame, (x, y), 15, color, -1)

    # Tampilkan teks status
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imshow("Virtual Mouse with UI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
