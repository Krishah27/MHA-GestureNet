import cv2

def start_webcam():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Healthcare Webcam Feed", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()