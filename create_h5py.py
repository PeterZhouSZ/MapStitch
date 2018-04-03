import glob
from io import BytesIO
from PIL import Image
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


# test = np.arange(9)#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# random.shuffle(test)
# print(test)
#
# data_dir = 'new_64/'
# shuffle_data = True
#
# size = int(re.search(r'\d+', data_dir).group()) #auto-extract integer from string
#
#
# files_in_directory = list(glob.glob(data_dir + '*.png'))
# num_images = len(files_in_directory)
# image_data = np.empty([num_images, size, size])
# coordinates = np.empty((len(files_in_directory), 2))
#
# idx = 0
#
#
# for filename in tqdm(files_in_directory):  # import all png
# 	image = np.array(Image.open(filename))[:, :, 0]
# 	image_data[idx] = image
#
# 	filename = filename.split('\\')[1]
# 	m = re.match(r'14_(.*),(.*)_terrain_(.*).png', filename)
# 	lat = float(m.group(1))
# 	lon = float(m.group(2))
# 	coordinates[idx] = (lat, lon)
#
# 	idx += 1
#
# # to shuffle data
# if shuffle_data:
# 	shuffled_indices = np.arange(num_images)
# 	#combined = list(zip(image_data, coordinates))
# 	random.shuffle(shuffled_indices)
#
# 	image_data = [image_data[i] for i in shuffled_indices]
# 	coordinates = [coordinates[i] for i in shuffled_indices]
# 	#shuffle(image_data)
#
# max_val = np.amax(image_data)
# min_val = np.amin(image_data)
# print('img range: {} {}'.format(min_val, max_val))
#
# hdf5_path = 'coastlines_binary_' + str(size) + '_images.hdf5'
# hdf5_file = h5py.File(hdf5_path, mode='w')
# imgs_shape = ( num_images, size, size ) #image_data.shape  #
# hdf5_file.create_dataset("images", imgs_shape, dtype='uint8')
# hdf5_file["images"][...] = image_data
#
# hdf5_file.create_dataset("coordinates", data=coordinates )
#
# hdf5_file.close()
# print('...finished creating hdf5')

#check contents of h5py
hdf5_path = 'coastlines_binary_128_images.hdf5'  # address to where you want to save the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# Total number of samples
images = hdf5_file["images"]
max_val = np.amax(images)
min_val = np.amin(images)
print('img range: {} {}'.format(min_val, max_val))

idx = np.random.randint(0, images.shape[0], size=16)
#idx = range(16)
#plt.figure()
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(14, 14), )
for ii, ax in zip(idx, axes.flatten()):
	ax.imshow(images[ii], aspect='equal', cmap='gray')  # [:,:,:]
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()