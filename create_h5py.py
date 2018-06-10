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


def subdivide(image, w_tiles, h_tiles, swidth, sheight):
	out = []
	for sh in range(h_tiles):
		for sw in range(w_tiles):
			subimg = image[sh * sheight:(sh + 1) * sheight, sw * swidth:(sw + 1) * swidth]
			out.append(subimg)

	return out

# test = np.arange(9)#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# random.shuffle(test)
# print(test)
#
data_dir = r'C:\\Users\\fruehsa\\Data\\ESRI\\11\\'
shuffle_data = True

#sizes = [128, 32, 16, 8]
size = 256 #int(re.search(r'\d+', data_dir).group()) #auto-extract integer from string

files_in_directory = list(glob.glob(data_dir + '*.jpg'))
random.shuffle(files_in_directory)

input_size = max(Image.open(files_in_directory[0]).size)

shrink = True
subdivide = False

num_images = len(files_in_directory)
print("{} files found in directory".format(num_images))

multiplier = 1
if subdivide:
	multiplier = (input_size // size) * (input_size // size)
	print("multiply by {} subimages".format(multiplier))
	num_images *= multiplier 

#if num_images > 1e6:
#num_images = np.min((num_images, 4999))	#restrict maximum number of images
#image_data = np.empty([num_images, size, size, 3])
#coordinates = np.empty((num_images, 2))

hdf5_path = 'data/esri_level_eleven_' + str(size) + 'x' + str(size) + '_1M_images.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')

#hdf5_file.create_dataset("train_img", train_shape, np.int8)


chunk_size = 1000
chunk_images_shape = (chunk_size, size, size, 3)
chunk_coordinates_shape = (chunk_size, 2)

imgs_shape = (0, size, size, 3) #image_data.shape  #
coords_shape = (0, 2) #image_data.shape  #
images_dataset = hdf5_file.create_dataset("images", imgs_shape, maxshape=(None, size, size, 3), dtype='uint8', chunks=chunk_images_shape)
coordinates_dataset = hdf5_file.create_dataset("coordinates", coords_shape, maxshape=(None, 2), chunks=chunk_coordinates_shape)

idx = 0
chunk_count = 0
chunk_images = np.empty(chunk_images_shape)
chunk_coordinates = np.empty(chunk_coordinates_shape)
for filename in tqdm(files_in_directory):  # import all png
	image = Image.open(filename)#[:, :, 0]

	filename = filename.split('\\')[-1]
	m = re.match(r'11_(.*)_(.*).jpg', filename)
	lat = float(m.group(1))
	lon = float(m.group(2))
		
	# if shrink:
	#image = image.resize([size, size], Image.ANTIALIAS)
	image = np.array(image)

	if len(set(image[:, :, 2].flatten())) < 10:  # too few colors in tile
		continue

	chunk_images[idx % chunk_size, :] = image[None]
	chunk_coordinates[idx % chunk_size, :] = (lat, lon)

	if idx % chunk_size == 0 and  idx > 0:
		images_dataset.resize((idx, size, size, 3))
		images_dataset[chunk_count:] = chunk_images

		coordinates_dataset.resize((idx, 2))
		coordinates_dataset[chunk_count:] = chunk_coordinates

		chunk_count += chunk_size

		# hdf5_file["images"][idx, ...] = image[None]
		# hdf5_file["coordinates"][idx, ...] = (lat, lon)
	
	#image_data[idx] = image
	#coordinates[idx] = (lat, lon)
	idx += 1

	# elif subdivide and multiplier > 1:
	# 	subimgs = subdivide(np.array(image), multiplier, multiplier, size, size)
	# 	for image in subimgs:
	# 		if len(set(image[:, :, 2].flatten())) < 10:  # too few colors in tile
	# 			continue

	# 		image_data[idx] = image
	# 		coordinates[idx] = (lat, lon)
	# 		idx += 1
	# else:
	# 	if len(set(image[:, :, 2].flatten())) < 10:  # too few colors in tile
	# 		continue
	# 	image = np.array(image)
	# 	image_data[idx] = image
	# 	coordinates[idx] = (lat, lon)
	# 	idx += 1

	if idx >= num_images:
		break

if idx % chunk_size != 0:
	last_idx = idx % chunk_size
	images_dataset.resize((idx, size, size, 3))
	images_dataset[chunk_count:] = chunk_images[:last_idx]

	coordinates_dataset.resize((idx, 2))
	coordinates_dataset[chunk_count:] = chunk_coordinates[:last_idx]

# if idx < num_images:
# 	image_data = image_data[:idx]
# 	coordinates = coordinates[:idx]
# 	num_images = image_data.shape[0]
# 	print("number of final images: {}".format(num_images))

# # to shuffle data
# if shuffle_data:
# 	shuffled_indices = np.arange(num_images)
# 	#combined = list(zip(image_data, coordinates))
# 	random.shuffle(shuffled_indices)
	
# 	image_data = [image_data[i] for i in shuffled_indices]
# 	coordinates = [coordinates[i] for i in shuffled_indices]

# max_val = np.amax(image_data)
# min_val = np.amin(image_data)
# print('img range: {} {}'.format(min_val, max_val))

#hdf5_file["images"][...] = image_data

#hdf5_file.create_dataset("coordinates", data=coordinates )

hdf5_file.close()
print('...finished creating hdf5')

#check contents of h5py
#size = 128
#hdf5_path = 'data/nasa_blue_marble_' + str(size) + '_images.hdf5'  # address to where you want to save the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# Total number of samples
images = hdf5_file["images"]
print(images.shape)
#sat_images = hdf5_file["satellite"]
# max_val = np.amax(images)
# min_val = np.amin(images)
# print('img range: {} {}'.format(min_val, max_val))

idx = np.random.randint(0, images.shape[0], size=36)
#idx = range(16)
#plt.figure()
fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(14, 14) )
i = 0
for ii, ax in zip(idx, axes.flatten()):
	#if i % 2 == 0:
	show = images[ii]
	# else:
	# 	show = sat_images[idx[int(i/2)]]
	#i+= 1
	ax.imshow(show, aspect='equal')  # [:,:,:]
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()