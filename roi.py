import cv2


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"

cap = cv2.VideoCapture(PATH + "videos/vid03.mp4")
total_frames = int(cap.get(7))
# cap.set(1, total_frames - 500)
cap.set(1, 100)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("select window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("select window", frame.shape[0], frame.shape[1])
r = cv2.selectROI("select window", frame)
print(r)
# cv2.imwrite(PATH + "scripts/roi.jpg", roi)
# cv2.imwrite(PATH + "scripts/frame.jpg", frame)