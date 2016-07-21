import cv2
import numpy as np

def main():
    img = cv2.imread('bee.png')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v_equ = cv2.equalizeHist(v)
    hsv_img_equ = cv2.merge([h, s, v_equ])
    bgr_img_equ = cv2.cvtColor(hsv_img_equ, cv2.COLOR_HSV2BGR)
    cv2.imwrite('opencv_v_equal.png', bgr_img_equ)


if __name__ == '__main__':
    main()