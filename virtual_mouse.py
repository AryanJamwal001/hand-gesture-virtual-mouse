import cv2
import mediapipe as mp
import pyautogui
import math

print("✅ Script started...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to access webcam.")
    exit()

print("✅ Webcam opened.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

def count_fingers(hand_landmarks):
    fingers = []

    # Tip landmarks: [Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)]
    tips_ids = [4, 8, 12, 16, 20]

    # Get y-position of MCP joints for comparison (for non-thumb fingers)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from webcam.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm_list = hand_landmarks.landmark

            # Index Finger Tip
            x = int(lm_list[8].x * screen_width)
            y = int(lm_list[8].y * screen_height)
            pyautogui.moveTo(x, y)

            # Thumb Tip
            thumb_x = int(lm_list[4].x * screen_width)
            thumb_y = int(lm_list[4].y * screen_height)

            # Draw circles
            cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)

            # Distance for pinch (click)
            distance = math.hypot(x - thumb_x, y - thumb_y)

            if distance < 40:
                print("🖱️ Click triggered")
                pyautogui.click()
                pyautogui.sleep(1)

            # Count fingers
            finger_count = count_fingers(hand_landmarks)

            if finger_count == 2:
                print("🖱️ Scroll Up")
                pyautogui.scroll(20)
                pyautogui.sleep(0.3)

            elif finger_count == 3:
                print("🖱️ Scroll Down")
                pyautogui.scroll(-20)
                pyautogui.sleep(0.3)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
