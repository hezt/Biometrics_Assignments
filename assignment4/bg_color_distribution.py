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
    pix_arr = np.array([[0.0] * 3] * count_int)
    pix_arr = np.float32(pix_arr)
    img_bg = [[[0.0] * 3] * width_int] * height_int
    img_bg = np.array(img_bg)
    img_bg = np.float32(img_bg)
    print(width_int, height_int, fps_int, count_int)

    fr_list = [] 
    for fr in range(0, count_int):
        _, img = cap.read()
        fr_list.append(np.float32(img)) 
    cap.release()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    for h in range(height_int):
        for w in range(width_int):
            for i in range(count_int):
                pix_arr[i][:] = fr_list[i][h][w][:]
            ret, label, center = cv2.kmeans(pix_arr, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            cnt=[0, 0]
        for m in range(0, len(label)):
            cnt[label[m][0]] += 1
        if cnt[0] > cnt[1]:
            img_bg[h][w] = center[0]
        else:
            img_bg[h][w] = center[1]

    cv2.imshow('bg_color_distribution' , np.uint8(img_bg))
    cv2.imwrite('bg_color_distribution.png', img_bg)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()