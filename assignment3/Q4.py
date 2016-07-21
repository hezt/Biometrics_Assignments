from PIL import Image, ImageOps
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt


def main():
    flower = Image.open("flower.jpg")

    plt.subplot(421)
    plt.imshow(flower, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('original')
    plt.subplot(422)
    flower = ImageOps.grayscale(flower)
    plt.imshow(flower, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('gray')

    aflower = np.asarray(flower)
    aflower = np.float32(aflower)
    U, S, Vt = linalg.svd(aflower)

    plt.subplot(423)
    k20 = fun(20, S, U, Vt)
    plt.imshow(k20, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('K = 20')

    plt.subplot(424)
    k50 = fun(50, S, U, Vt)
    plt.imshow(k50, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('K = 50')

    plt.subplot(425)
    k100 = fun(100, S, U, Vt)
    plt.imshow(k100, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('K = 100')

    plt.subplot(426)
    k200 = fun(200, S, U, Vt)
    plt.imshow(k200, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('K = 200')


    plt.subplot(427)
    plt.plot(S, 'b.')
    plt.savefig('Q4_result.ps')


    plt.show()


def fun(K, S, U, Vt):
    Sk = np.diag(S[:K])
    Uk = U[:, :K]
    Vtk = Vt[:K, :]
    aImk = np.dot(Uk, np.dot(Sk, Vtk))
    Imk = Image.fromarray(aImk)
    return Imk

if __name__ == '__main__':
    main()
