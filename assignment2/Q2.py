import numpy as np
import scipy as sp
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def rotate():
    '''
    In the image, rotate a rectangular region by 45 degree
    counter-clockwise, whose ver- tices are (100,100),
    (100,400),(400,100),(400,400).
    '''
    img = Image.open('lena.png')
    img_croped = img.crop((100, 100, 400, 400)).rotate(45)
    img.paste(img_croped, (100, 100))
    img.save('rotate.png')


def histogram_equalization():
    '''
    Perform histogram equalization on lena.png.
    Use matplotlib to plot the histogram figure for
    both original image and processed image
    '''
    img = Image.open('lena.png')
    plt.subplot(221)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(222)
    img_array = np.array(img)
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img_array.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf','histogram'), loc = 'upper left')

    plt.subplot(224)
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    img_array = np.array(img2)
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img_array.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf','histogram'), loc = 'upper left')

    plt.subplot(223)
    plt.imshow(img2, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('hist_equal.png')
    plt.show()


def filtering():
    '''
    Perform Max Filtering, Min Filtering, and Median Filter on lena.png.
    '''
    img = Image.open('lena.png')
    maxf_img = img.filter(ImageFilter.MaxFilter) 
    maxf_img.save('max_filtering.png')
    minf_img = img.filter(ImageFilter.MinFilter) 
    minf_img.save('min_filtering.png')
    medianf_img = img.filter(ImageFilter.MedianFilter) 
    medianf_img.save('median_filtering.png')

def gaussian_blur():
    '''
    Perform Gaussian Blur with sigma equal to 3 and 5.
    '''
    img = Image.open('lena.png')
    gb3_img = img.filter(ImageFilter.GaussianBlur(radius=3))
    gb5_img = img.filter(ImageFilter.GaussianBlur(radius=5))
    gb3_img.save('gaussian_blur_sigma_3.png')
    gb5_img.save('gaussian_blur_sigma_5.png')

def main():
    rotate()
    histogram_equalization()
    filtering()
    gaussian_blur()


if __name__ == '__main__':
    main()
