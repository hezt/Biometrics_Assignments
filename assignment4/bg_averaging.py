import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture('traffic.mp4')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width_int = int(width)
    height_int = int(height)
    fps_int = int(fps)
    count_int = int(count)
    print(width_int, height_int, fps_int, count_int)

    _, img = cap.read()
    img_avg = np.float32(img)
    for fr in range(1, count_int):
        _, img = cap.read()
        img_avg += np.float32(img)
    img_avg /= count_int
    img_avg = np.uint8(img_avg) 
    cv2.imshow('bg_averaging', img_avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    cv2.imwrite('bg_averaging.png', img_avg)


if __name__ == '__main__':
    main()
