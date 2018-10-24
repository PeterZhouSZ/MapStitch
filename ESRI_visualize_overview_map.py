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

level = 7
data_dir = 'D:\\Data\\ESRI\\' + str(level) + '\\'

#files_in_directory = list(glob.glob(data_dir + '*.jpg'))

input_size = 256# max(Image.open(files_in_directory[0]).size)
t_size = 4

power_of_two = 2 ** level

#num_images = len(files_in_directory)
#print("{} files of size {}x{} found in directory".format(num_images, input_size, input_size))

ones = np.ones((t_size*power_of_two, t_size*power_of_two), dtype='uint8')
average_output_image = np.stack((186*ones, 225*ones, 242*ones), axis=2)

for i in tqdm(range(power_of_two*power_of_two)):#*power_of_two)):  
    y = i % power_of_two
    x = int(i / power_of_two)
    
    filename_split = str(level) + '_' + str(y) + '_' + str(x) + '.jpg'
    filename = (data_dir + filename_split)

    try:    
        image = Image.open(filename) 
        image.thumbnail((t_size, t_size))   
        average_output_image[y * t_size:(y + 1) * t_size, x * t_size:(x + 1) * t_size] = image
    except FileNotFoundError:
        #print("File {} not found, skipping tile".format(filename_split) ) 
        continue
            
out_img = Image.fromarray(average_output_image)
out_img.save('D:\\Data\\ESRI\\' + str(level) + '_average.jpg')