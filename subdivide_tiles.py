#!/usr/bin/python

import glob
from io import BytesIO
from PIL import Image
import os, sys
from tqdm import tqdm

import time
import random
import cv2
import numpy as np
import re
import h5py
import matplotlib.pyplot as plt

#WATER_MIN_TERRAIN = np.array([0, 0, 0], dtype=np.uint8) #minimum value of black pixel in RGB order
#WATER_MAX_TERRAIN = np.array([5, 5, 5], dtype=np.uint8) #maximum value of black pixel in RGB order

WATER_MIN_TERRAIN = np.array([155, 195, 227], dtype=np.uint8) #minimum value of blue pixel in RGB order
WATER_MAX_TERRAIN = np.array([220, 240, 255], dtype=np.uint8) #maximum value of blue pixel in RGB order

def resize_tile(tile, width, height):
	return tile.resize((width, height), Image.ANTIALIAS)  

def subdivide(image, swidth, sheight):
	width, height = image.shape

	nwidth = int(np.floor(width / swidth))
	nheight = int(np.floor(height / sheight))
	#hwidth = round(width / 2)
	#hheight = round(height / 2)
	#swidth = round(width / num)
	#sheight = round(height / num)

	out = []
	for sw in range(nwidth):
		for sh in range(nheight):
			subimg = image[sh*sheight:(sh+1)*sheight,sw*swidth:(sw+1)*swidth]
			#box = (sw*swidth, sh*sheight, (sw+1)*swidth, (sh+1)*sheight)
			#subimg = image.crop(box)
			out.append(subimg)
	# box_1 = (0, 0, hwidth, hheight)
	# box_2 = (0, hheight, hwidth, height)
	# box_3 = (hwidth, 0, width, hheight)
	# box_4 = (hwidth, hheight, width, height)
	# sub_1 = image.crop(box_1)
	# sub_2 = image.crop(box_2)
	# sub_3 = image.crop(box_3)
	# sub_4 = image.crop(box_4)

	return out #[sub_1, sub_2, sub_3, sub_4]
	
def convert_tile_to_binary(npimage, mask):
	npimage[np.where(mask == [000])] = [255, 255, 255] 	  # blue image parts turn black
	npimage[np.where(mask == [255])] = [  0,   0,   0]   # everything else turns white

	#convert back to numpy image
	return npimage[:, :, 0] #Image.fromarray(npimage)

#remove lake and river tiles	
def strictly_coastline(mask, size):
	width, height = size
	
	edges = [mask[0,:], mask[1:,-1], mask[-1,:-1], mask[1:-1,0]]
	edges_only = np.concatenate(edges)
	water_percent = np.count_nonzero(edges_only) / (2 * (width + height))
	#print('sc: ', water_percent)
	return water_percent > 0.05

def create_dir(name):		
	try:
		os.makedirs(name)
	except OSError as e:
		print('Directory already exists!')
		
if __name__ == "__main__":
	image_list = []
	#filename_list = []
	directory_out = os.path.join(os.getcwd(), 'coastlines_binary_128') #'coastlines_terrain_square') #
	
	#create_dir(directory_128)
	
	files = list(glob.iglob('coastlines_terrain_square/*.png' ))

	#size = 128
	#subdiv = 5
	sizes   = [256]#128, 64, 32] #[256, 128, 64, 32]
	subdivs = [2]# 4,  8, 16] #[2, 5, 10, 20]


	for i in range(len(sizes)):
		size = sizes[i]
		subdiv = subdivs[i]

		all_water = np.zeros((size, size))
		all_land = np.ones((size, size))

		multiplier = subdiv*subdiv
		shuffle_data = True
		num_images = len(files) * multiplier

		image_data = np.empty([num_images, size, size])
		coordinates = np.empty((num_images, 2))

		print('{} images found...'.format(num_images))

		image_data[0] = all_water
		image_data[1] = all_land

		idx = 2
		all_equal = 0
		for filename in tqdm(files): #glob.glob('coastlines_terrain_14/*.png'): #assuming png format
			#if idx > 1000:
		    #		break

			image = Image.open(filename)
			cimage = image.convert("RGB")
			npimage = np.array(cimage)

			blue_mask = cv2.inRange(npimage, WATER_MIN_TERRAIN, WATER_MAX_TERRAIN)
			grayscale = convert_tile_to_binary(npimage, blue_mask)

			#grayscale = grayscale.resize((512, 512), Image.ANTIALIAS)
			#print(image.size)
			subimgs = subdivide(grayscale, size, size)

			filename = filename.split('\\')[1]
			m = re.match(r'14_(.*),(.*)_terrain.png', filename)
			lat = float(m.group(1))
			lon = float(m.group(2))

			for subidx, subimg in enumerate(subimgs):
				#gray_im = Image.fromarray(subimg)

				#filename_complete = os.path.join(directory_out, filename)
				#subimg.save(filename_complete)
				if len(set(subimg.flatten())) <= 1: # all zeros or all ones
					all_equal += 1
					continue

				#image_data[idx * multiplier + subidx] = subimg
				#coordinates[idx * multiplier + subidx] = (lat, lon)
				image_data[idx] = subimg
				coordinates[idx] = (lat, lon)
				idx += 1

			image.close()

		#unique_images, unique_indices = np.unique(image_data, axis=0, return_index=True)
		#unique_coordinates = coordinates[unique_indices]

		#image_data = unique_images
		#coordinates = unique_coordinates
		image_data = image_data[0:idx]
		coordinates = coordinates[0:idx]
		num_images = image_data.shape[0]

		print('did not consider {} all land/water images in dataset...'.format(all_equal))

		# to shuffle data
		if shuffle_data:
			shuffled_indices = np.arange(num_images)
			# combined = list(zip(image_data, coordinates))
			random.shuffle(shuffled_indices)

			image_data = [image_data[i] for i in shuffled_indices]
			coordinates = [coordinates[i] for i in shuffled_indices]
		# shuffle(image_data)

		max_val = np.amax(image_data)
		min_val = np.amin(image_data)
		print('img range: {} {}'.format(min_val, max_val))

		hdf5_path = 'data/coastlines_binary_cleaned_' + str(size) + '_images.hdf5'
		hdf5_file = h5py.File(hdf5_path, mode='w')
		imgs_shape = (num_images, size, size)  # image_data.shape  #
		if not "images" in hdf5_file: #creating new hdf5 file
			#hdf5_file.create_dataset("images", data=image_data)
			hdf5_file.create_dataset("images", imgs_shape, dtype='uint8')
			hdf5_file["images"][...] = image_data

			hdf5_file.create_dataset("coordinates", data=coordinates)
		else: #append (since file exists already)
			hdf5_file["images"].resize((hdf5_file["images"].shape[0] + num_images), axis=0)
			hdf5_file["images"][-hdf5_file["images"].shape[0]:] = image_data

			hdf5_file["coordinates"].resize((hdf5_file["coordinates"].shape[0] + num_images), axis=0)
			hdf5_file["coordinates"][-hdf5_file["coordinates"].shape[0]:] = coordinates


		print('...file contains {} images now'.format(hdf5_file["images"].shape[0]))
		hdf5_file.close()
		print('...finished creating hdf5')
