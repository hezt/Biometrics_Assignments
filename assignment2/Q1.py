import numpy as np
from scipy import signal

def main():
    matrix_a = np.array([
        [1, 3, 2, 4],
        [2, 2, 3, 4],
        [5, 5, 4, 5],
        [8, 9, 0, 1]])
    matrix_b = np.array([
        [1, 2, 3, 4],
        [2, 1, 3, 0],
        [4, 1, 3, 4],
        [2, 4, 3, 4]])
    matrix_result = signal.convolve(matrix_a, matrix_b)
    print(matrix_result)

if __name__ == '__main__':
    main()
