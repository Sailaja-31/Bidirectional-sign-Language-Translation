
import cv2
import mediapipe as mp
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

GESTURE_IMAGES = {
    "THANK YOU": "thankyou.png",
    "HAPPY": "happy.png",
    "SAD": "sad.png",
    "APPRECIATE": "appreciate.png",
    "GREAT": "great.png"
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --------------------------------------------------
# FINGER DETECTION
# --------------------------------------------------

def finger_open(lm, tip, pip):
    # For a roughly upright hand, fingertips above joints
    # indicate open fingers.
    return lm[tip].y < lm[pip].y


def thumb_open(lm):
    # Approximate thumb extension using landmark distances.
    # This may need adjustment for different hand orientations.
    tip_distance = abs(lm[4].x - lm[5].x)
    joint_distance = abs(lm[3].x - lm[5].x)
    return tip_distance > joint_distance


def recognize_gesture(lm):
    thumb = thumb_open(lm)
    index = finger_open(lm, 8, 6)
    middle = finger_open(lm, 12, 10)
    ring = finger_open(lm, 16, 14)
    pinky = finger_open(lm, 20, 18)

   # SUPER: thumb/index closed; middle/ring/little open
    if not index and middle and ring and pinky and not thumb:
      return "SUPER"

# CHEERS: index and middle open; ring and little closed
    elif index and middle and not ring and not pinky:
      return "CHEERS"

# HELLO: all four fingers and thumb open
    elif index and middle and ring and pinky and thumb:
     return "HELLO"

# STOP: four fingers open and thumb closed
    elif index and middle and ring and pinky and not thumb:
     return "STOP"

# YES: four fingers closed and thumb extended
    elif not index and not middle and not ring and not pinky and thumb:
     return "YES"

    return "Gesture not recognized"


# --------------------------------------------------
# WEBCAM HELPER
# --------------------------------------------------

def open_camera():
    # Try DirectShow first on Windows, then default backend.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test whether a frame can be read.
            success, frame = camera.read()
            if success and frame is not None:
                return camera

        camera.release()

    return None


# --------------------------------------------------
# OPTION 1: SIGN GESTURE TO TEXT
# --------------------------------------------------

def sign_to_text():
    camera = open_camera()

    if camera is None:
        print("\nERROR: Webcam could not be opened.")
        print("Close the Windows Camera app, Zoom, Teams, etc.")
        print("Check Windows Settings > Privacy > Camera.")
        input("Press Enter to return to menu...")
        return

    print("\nWebcam started. Press Q to return to menu.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                text = "Show your hand"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        text = recognize_gesture(
                            hand_landmarks.landmark
                        )

                        # Display gesture name
                        cv2.putText(
                            frame,
                            text,
                            (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

                        # Display only the first detected hand's result
                        break

                else:
                    cv2.putText(
                        frame,
                        text,
                        (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow("Sign Gesture to Text", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Returned from webcam mode.")


# --------------------------------------------------
# OPTION 2: TEXT TO SIGN GESTURE IMAGE
# --------------------------------------------------

def text_to_sign():
    print("\nAvailable words:")
    for word in GESTURE_IMAGES:
        print("-", word)

    word = input("\nEnter a word: ").strip().upper()

    if word not in GESTURE_IMAGES:
        print("Word not available in the image list.")
        return

    image_path = BASE_DIR / GESTURE_IMAGES[word]

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        print("Place the image in the same folder as hand_test.py.")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not open this image.")
        return

    cv2.imshow(f"Sign Gesture: {word}", image)
    print("Press any key while the image window is active to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------

def main():
    while True:
        print("\n==========================================")
        print("   SIGN LANGUAGE RECOGNITION SYSTEM")
        print("==========================================")
        print("1. Sign Gesture to Text")
        print("2. Text to Sign Gesture")
        print("3. Exit")
        print("==========================================")

        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            sign_to_text()

        elif choice == "2":
            text_to_sign()

        elif choice == "3":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
