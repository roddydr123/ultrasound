import cv2
import sys


PATH = "/home/david/Documents/uni/year-5/ultrasound/"

cap = cv2.VideoCapture(f"{PATH}videos/{sys.argv[1]}.mp4")
total_frames = int(cap.get(7))
# cap.set(1, total_frames - 500)
cap.set(1, 100)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("select window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("select window", frame.shape[0], frame.shape[1])
r = cv2.selectROI("select window", frame)
print(f"{r[0]},{r[1]},{r[2]},{r[3]}")
# cv2.imwrite(PATH + "scripts/roi.jpg", roi)
# cv2.imwrite(PATH + "scripts/frame.jpg", frame)
