import cv2
import numpy


PATH = "C:\\Users\\david\\Documents\\uni\\year-5\\ultrasound\\photos"


def main():
    img = cv2.imread(PATH + "Image11.jpg")
    cv2.imwrite("analysed/output.jpg", img)


main()