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


data_dir = 'new_64/'
size = int(re.search(r'\d+', data_dir).group()) #auto-extract integer from string


files_in_directory = list(glob.glob(data_dir + '*.png'))
num_images = len(files_in_directory)
image_data = np.empty([num_images, size, size])
idx = 0

for filename in tqdm(files_in_directory):  # import all png
	image = np.array(Image.open(filename))[:, :, 0]
	image_data[idx] = image
	idx += 1

shuffle_data = False

# to shuffle data
if shuffle_data:
	shuffle(image_data)

max_val = np.amax(image_data)
min_val = np.amin(image_data)
print('img range: {} {}'.format(min_val, max_val))

hdf5_path = 'coastlines_binary_' + str(size) + '_images.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
imgs_shape = image_data.shape  # (num_imgs, 64, 64)
hdf5_file.create_dataset("images", imgs_shape, dtype='uint8')
hdf5_file["images"][...] = image_data

hdf5_file.close()
print('...finished creating hdf5')


#check contents of h5py
hdf5_path = 'coastlines_binary_64_images.hdf5'  # address to where you want to save the hdf5 file
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