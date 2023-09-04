import subprocess
import cv2
import numpy as np

def main():
    # Start capturing the live preview using gphoto2
    capture_process = subprocess.Popen(
        ['gphoto2', '--stdout', '--capture-movie'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    # Create a VideoCapture object to read the live preview
    cap = cv2.VideoCapture('/dev/video0')  # Specify the correct video device path

    if not cap.isOpened():
        print("Failed to open capture process")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the live preview
        cv2.imshow('Live View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    capture_process.kill()

if __name__ == "__main__":
    main()
