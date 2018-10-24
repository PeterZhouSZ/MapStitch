import glob
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

level = 11
data_dir = 'D:\\Data\\SAT\\SAT_train_HR\\'
save_dir = 'D:\\Data\\SAT\\SAT_train_LR_bicubic\\'

files_in_directory = list(glob.glob(data_dir + '*.jpg'))

input_size = max(Image.open(files_in_directory[0]).size)
input_size_2x = int(input_size / 2)
input_size_3x = int(input_size / 3)
input_size_4x = int(input_size / 4)

idx = 1
for filename in tqdm(files_in_directory):
    filename_split = filename.split('\\')[-1]
    
    img_idx = str(idx).zfill(4)
    os.rename(data_dir + filename_split, data_dir + img_idx + '.jpg')
    os.rename(save_dir + 'X2\\' + filename_split, save_dir + 'X2\\' + img_idx + 'x2.jpg')
    os.rename(save_dir + 'X3\\' + filename_split, save_dir + 'X3\\' + img_idx + 'x3.jpg')
    os.rename(save_dir + 'X4\\' + filename_split, save_dir + 'X4\\' + img_idx + 'x4.jpg')
    idx += 1
    
    
