import cv2
import numpy as np
from scipy import signal
from PIL import Image

def main():
    img = cv2.imread('lena.png', 0)
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ])
    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
        ])
    img_arr = np.array(img)
    Mx = signal.convolve2d(img_arr, Gx)
    My = signal.convolve2d(img_arr, Gy)
    M = np.sqrt(Mx * Mx + My * My)
    M = 255 * M / np.max(M)
    M = np.array(M, dtype='uint8')
    img =  Image.fromarray(M)
    img.show() 
    img.save('edge_detection.png')

if __name__ == '__main__':
    main()