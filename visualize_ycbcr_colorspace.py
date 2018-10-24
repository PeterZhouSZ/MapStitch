import glob
import math
from io import BytesIO
from PIL import Image, ImageDraw
import os, sys
from tqdm import tqdm
import h5py
import time
import random
import cv2
import numpy as np
import os.path
import re
import matplotlib
import matplotlib.pyplot as plt
from shutil import copyfile

def rgb2ycbcr(im):
    xform = np.array([[ 0.2989,  0.5866,  0.1145], [-0.1688, -0.3312,  0.5000], [ 0.5000, -0.4184, -0.0816]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[ 1.0000, -0.0010,  1.4020], [ 1.0000, -0.3441, -0.7140], [ 1.0000,  1.7718,  0.0010]])
    ycbcr = im.astype(np.float)
    ycbcr[:, :, [1, 2]] -= 128
    rgb = ycbcr.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

# def rgb2ycbcg(im):
#     xform = np.array([[ 1,  1,  1], [1, -1,  0], [ 1, 0, -1]])
#     ycbcg = im.dot(xform.T)
#     #ycbcr[:, :, [1, 2]] += 128
#     return np.uint8(ycbcg)
#
# def ycbcg2rgb(im):
#     xform = np.array([[ 1,  1,  1], [1, -2,  0], [1, 0,  -2]])
#     ycbcg = im.astype(np.float)
#     ycbcg /= np.array([ 3,  2,  2])
#     #ycbcg[:, :, [1, 2]] -= 128
#     rgb = ycbcg.dot(xform.T)
#     np.putmask(rgb, rgb > 255, 255)
#     np.putmask(rgb, rgb < 0, 0)
#     return np.uint8(rgb)

level = 11
data_dir = 'D:\\Data\\ESRI\\9_land_only\\'

files_in_directory = list(glob.glob(data_dir + '*.jpg'))

input_size = max(Image.open(files_in_directory[0]).size)
t_size = 4

power_of_two = 2 ** level

select_num = 27
select_subset = random.sample(files_in_directory, select_num)

columns = 6 # * math.floor(math.sqrt(select_num))
rows = math.ceil(select_num / 2)#math.ceil(select_num / columns)
fig=plt.figure(figsize=(columns*3, rows*3))

idx = 1
for i in tqdm(range(select_num)):
    img = Image.open(select_subset[i], 'r')
    npImg = np.array(img)

    ycbcrImg = rgb2ycbcr(npImg)

    for channel in range(3):
        s = np.setxor1d([0, 1, 2], [channel])

        ycbcr_channel = np.array(ycbcrImg, copy=True)
        ycbcr_channel[:, :, s] = 128
        # convert back to RGB
        rgb_subplot = ycbcr2rgb(ycbcr_channel)

        fig.add_subplot(rows, columns, idx)
        plt.axis('off')
        plt.imshow(rgb_subplot)
        idx += 1


plt.axis('off')
plt.show()