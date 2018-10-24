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

data_dir = 'D:\\Data\\SAT\\SAT_train_HR\\'
save_dir = 'D:\\Data\\SAT\\SAT_train_LR_bicubic\\'

ext = '.jpg'
files_in_directory = list(glob.glob(data_dir + '*' + ext))

input_size = max(Image.open(files_in_directory[0]).size)

output_multipliers = [8]#[2, 3, 4, 8]

for filename in tqdm(files_in_directory):
    filename_split = filename.split('\\')[-1]
    filename_no_ext = filename_split.split('.')[0]

    image = Image.open(filename)

    for multiplier in output_multipliers:
        output_size = int(input_size / multiplier)
    #
    # image_2x = image.resize((input_size_2x, input_size_2x), Image.BICUBIC)
    # image_2x.save(save_dir + '\\X2\\' + filename_split)
    #
    # image_3x = image.resize((input_size_3x, input_size_3x), Image.BICUBIC)
    # image_3x.save(save_dir + '\\X3\\' + filename_split)
    #
    # image_4x = image.resize((input_size_4x, input_size_4x), Image.BICUBIC)
    # image_4x.save(save_dir + '\\X4\\' + filename_split)

        image_resized = image.resize((output_size, output_size), Image.BICUBIC)
        image_resized.save(save_dir + '\\X' + str(multiplier) + '\\' + filename_no_ext + 'x' + str(multiplier) + ext)

