#数据扩充

from PIL import Image
import os
import numpy as np
import tensorflow as tf
import PIL.Image as img
import matplotlib.pyplot as plt

img_path = r'./test_data'
resize_path = r'./resize_image'

"""
#将图片设成统一大小
for i in os.listdir(img_path):
    im =Image.open(os.path.join(img_path, i))
    out = im.resize((128, 128))
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    out.save(os.path.join(resize_path, i))
"""

"""
#图像基本变换
filelist = os.listdir(img_path)
total_num = len(filelist)
print(total_num)
for subdir in filelist:
    sub_dir = img_path + '/' + subdir
    im = img.open(sub_dir)
    # ng=im.transpose(img.FLIP_TOP_BOTTOM) #上下对换。
    # ng=im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
    # ng=im.transpose(Image.ROTATE_270) #顺时针旋转。
    ng = im.rotate(45)  # 逆时针旋转。
    ng.save(resize_path + '/' + subdir)

"""

"""
#裁剪
def imgCrop(im, x, y, w, h):
    img = im[x:np.clip(x+w,0,im.shape[0]),y:np.clip(y+h,0,im.shape[1])]
    return img

im = img.imread(os.path.join(img_path, i))
out = imgCrop(im,20,10,250,90)
if not os.path.exists(resize_path):
    os.makedirs(resize_path)
img.imsave(os.path.join(resize_path, i), out)
"""


"""
#灰度
def imgGray(im):
    imgarray = np.array(im)
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(rows):
     for j in range(cols):
       imgarray[i, j, :] = (imgarray[i, j, 0] * 0.299 + imgarray[i, j, 1] * 0.587 + imgarray[i, j, 2] * 0.114)
     return imgarray

for i in os.listdir(img_path): 
   im = img.imread(os.path.join(img_path, i))
   out = imgGray(im)
   if not os.path.exists(resize_path):
    os.makedirs(resize_path)
   img.imsave(os.path.join(resize_path, i),out)
"""

"""
#二值化
def imgThreshold(im, threshold):
    imgarray = np.array(im)
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(rows):
     for j in range(cols):
      gray = (imgarray[i, j, 0] * 0.299 + imgarray[i, j, 1] * 0.587 + imgarray[i, j, 2] * 0.114)
      if gray <= threshold :
       imgarray[i,j,:] = 0
      else:
       imgarray[i,j,:] = 255
    return imgarray

for i in os.listdir(img_path): 
   im = img.imread(os.path.join(img_path, i))
   out = imgThreshold(im, 128)
   if not os.path.exists(resize_path):
    os.makedirs(resize_path)
   img.imsave(os.path.join(resize_path, i),out)
"""


#图像高斯模糊
def gausskernel(radius, sigma):
    length = 2 * radius + 1
    kernel = np.zeros(length)
    print("kernel size: ", str(kernel.shape))
    sum = 0.0
    for i in range(length):
        kernel[i] = float(np.exp(-(i - radius) * (i - radius) / (2.0 * sigma * sigma)))
        sum += kernel[i]
    for i in range(length):
        kernel[i] = kernel[i] / sum
    return kernel
def imgGaussFilter(im, sigma):
    """
    Gauss filter.
        """
    imarray = np.array(im)
    res = np.array(im)
    radius = sigma
    kernel = gausskernel(radius, sigma*3.0)
    print(str(kernel))
    tempb = 0.0
    tempg = 0.0
    tempr = 0.0
    rem = 0.0
    t = 0.0
    v = 0.0
    K = 0.0
    rows = im.shape[0]
    cols = im.shape[1]
    for y in range(rows):
        for x in range(cols):
            tempb = 0.0
            tempg = 0.0
            tempr = 0.0
            for k in range(-radius, radius + 1):
                rem = np.abs(x + k) % cols
                K = kernel[k+radius]
                tempr = tempr + imarray[y,rem,0] * K
                tempg = tempg + imarray[y,rem,1] * K
                tempb = tempb + imarray[y,rem,2] * K
            res[y,x,0] = tempr
            res[y,x,1] = tempg
            res[y,x,2] = tempb
    for x in range(cols):
        for y in range(rows):
            tempb = 0.0
            tempg = 0.0
            tempr = 0.0
            for k in range(-radius, radius + 1):
                rem = np.abs(y + k) % rows
                K = kernel[k+radius]
                tempr = tempr + res[rem,x,0] * K
                tempg = tempg + res[rem,x,1] * K
                tempb = tempb + res[rem,x,2] * K
            imarray[y,x,0] = tempr
            imarray[y,x,1] = tempg
            imarray[y,x,2] = tempb
    return imarray

for i in os.listdir(img_path):
   im = img.imread(os.path.join(img_path, i))
   out = imgGaussFilter(im, 10)
   if not os.path.exists(resize_path):
    os.makedirs(resize_path)
   img.imsave(os.path.join(resize_path, i),out)