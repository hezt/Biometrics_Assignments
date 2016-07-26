import numpy as np
from matplotlib import pyplot as plt


def main():
    px = 3
    py = 3
    plt.scatter(px, py)
    plt.annotate('original' + ' x: ' + str(round(px, 2)) + ' y: ' + str(round(py, 2)), (px, py))
    cx = 2
    cy = 2
    t1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
        ])
    
    t2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
        ])
    for i in range(1, 8):
        theta = 1/4 * np.pi * i 
        r = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])
        result = np.dot(np.dot(np.dot(t2, r), t1), np.array([px, py, 1]))
        px_new = result[0] / result[2]
        py_new = result[1] / result[2]
        plt.scatter(px_new, py_new)
        plt.annotate(str(i) + '*1/4*pi' + ' x: ' + str(round(px_new, 2)) + ' y: ' + str(round(py_new, 2)), (px_new, py_new))
    plt.title('Geometric Transform')
    plt.savefig('geometric_transform.ps')
    plt.show()

if __name__ == '__main__':
    main()