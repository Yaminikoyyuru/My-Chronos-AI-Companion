import math

import cv2


def draw_full_robot(screen, state, t, talking, message, condition):
    if state == "EMERGENCY":
        theme_color = (0, 0, 255)
    elif state == "NORMAL_COMPANION":
        theme_color = (0, 255, 0)
    else:
        theme_color = (255, 192, 0)

    glow = int(12 * math.sin(t * 15)) if state == "EMERGENCY" else int(6 * math.sin(t * 4))

    # Antenna & Head
    cv2.line(screen, (320, 110), (320, 70), (150, 150, 150), 4)
    cv2.circle(screen, (320, 65), max(1, 10 + (glow if state == "EMERGENCY" else 0)), theme_color, -1)
    cv2.rectangle(screen, (240, 110), (400, 210), (80, 80, 80), -1)

    # Eyes
    if math.sin(t * 2.5) > 0.96 and state != "EMERGENCY":
        cv2.line(screen, (265, 150), (295, 150), theme_color, 4)
        cv2.line(screen, (345, 150), (375, 150), theme_color, 4)
    else:
        cv2.circle(screen, (280, 150), 16, theme_color, -1)
        cv2.circle(screen, (360, 150), 16, theme_color, -1)

    # Mouth
    if talking:
        mouth_open = int(14 * abs(math.sin(t * 22))) + 4
        cv2.rectangle(screen, (290, 185 - mouth_open // 2), (350, 185 + mouth_open // 2), theme_color, -1)
    elif state == "EMERGENCY":
        cv2.ellipse(screen, (320, 195), (20, 10), 0, 0, 360, theme_color, 3)
    elif state == "NORMAL_COMPANION":
        cv2.ellipse(screen, (320, 175), (25, 15), 0, 0, 180, theme_color, 3)
    else:
        cv2.line(screen, (295, 185), (345, 185), theme_color, 3)

    # Chest body & wave
    cv2.rectangle(screen, (300, 210), (340, 230), (120, 120, 120), -1)
    cv2.rectangle(screen, (200, 230), (440, 390), (60, 60, 60), -1)
    if talking or state == "NORMAL_COMPANION":
        for i in range(6):
            wave_h = int(25 * math.sin(t * 12 + i * 1.5))
            cv2.line(screen, (250 + i * 28, 310 - wave_h), (250 + i * 28, 310 + wave_h), theme_color, 3)
    else:
        cv2.line(screen, (250, 310), (390, 310), (100, 100, 100), 2)

    # Arms & Legs
    arm_swing = int(6 * math.sin(t * 3.5))
    cv2.rectangle(screen, (160, 240 + arm_swing), (195, 340 + arm_swing), (80, 80, 80), -1)
    cv2.rectangle(screen, (445, 240 - arm_swing), (480, 340 - arm_swing), (80, 80, 80), -1)
    cv2.rectangle(screen, (250, 390), (280, 450), (100, 100, 100), -1)
    cv2.rectangle(screen, (360, 390), (390, 450), (100, 100, 100), -1)

    # Subtitle HUD
    cv2.rectangle(screen, (15, 420), (625, 470), (20, 20, 20), -1)
    short_msg = message[:52] + "..." if len(message) > 52 else message
    cv2.putText(screen, f"Robot: {short_msg}", (25, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(screen, f"PATIENT: {condition}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(screen, f"MODE: {state}", (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme_color, 1)