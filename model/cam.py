import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't load webcam.\n")
    else:
        print("Success.\n")
        cap.release()

if __name__ == "__main__":
    main()
