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

WATER_MIN_TERRAIN = np.array([155, 195, 230], dtype=np.uint8) #minimum value of blue pixel in RGB order
WATER_MAX_TERRAIN = np.array([215, 240, 255], dtype=np.uint8) #maximum value of blue pixel in RGB order

def resize_tile(tile, width, height):
	return tile.resize((width, height), Image.ANTIALIAS)  

def subdivide(image, swidth, sheight):
	width, height = image.size

	nwidth = int(np.floor(width / swidth))
	nheight = int(np.floor(height / sheight))
	#hwidth = round(width / 2)
	#hheight = round(height / 2)
	#swidth = round(width / num)
	#sheight = round(height / num)

	out = []
	for sw in range(nwidth):
		for sh in range(nheight):
			box = (sw*swidth, sh*sheight, (sw+1)*swidth, (sh+1)*sheight)
			subimg = image.crop(box)
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
	npimage[np.where(blue_mask == [000])] = [255,255,255] 	# blue image parts turn black
	npimage[np.where(blue_mask == [255])] = [000,000,000]   # everything else turns white

	#convert back to numpy image
	return npimage#Image.fromarray(npimage)

#remove lake and river tiles	
def strictly_coastline(mask, size):
	width, height = size
	
	edges = [mask[0,:], mask[1:,-1], mask[-1,:-1], mask[1:-1,0]]
	edges_only = np.concatenate(edges)
	water_percent = np.count_nonzero(edges_only) / (2 * (width + height))
	#print('sc: ', water_percent)
	return water_percent > 0.05

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_dir(name):		
	try:
		os.makedirs(name)
	except OSError as e:
		print('Directory already exists!')
		
if __name__ == "__main__":
	image_list = []
	#filename_list = []
	#directory_out = os.path.join(os.getcwd(), 'new_64') #'coastlines_binary_128')
	
	#create_dir(directory_128)
	
	files = list(glob.iglob('coastlines_terrain_14/*.png' ))

	size = 64
	subdiv = 10
	multiplier = subdiv*subdiv
	shuffle_data = True
	num_images = len(files) * multiplier

	image_data = np.empty([num_images, size, size])
	coordinates = np.empty((num_images, 2))

	print('{} images found...'.format(num_images))
	#printProgressBar(0, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)
	
	for idx, filename in tqdm(enumerate(files)): #glob.glob('coastlines_terrain_14/*.png'): #assuming png format
		image = Image.open(filename)
		image = image.resize((640, 640), Image.ANTIALIAS)

		#print(image.size)
		subimgs = subdivide(image, size, size)

		filename = filename.split('\\')[1]
		m = re.match(r'14_(.*),(.*)_terrain.png', filename)
		lat = float(m.group(1))
		lon = float(m.group(2))

		for subidx, subimg in enumerate(subimgs):
			cimage = subimg.convert("RGB")
			npimage = np.array(cimage)
			#mask blue parts in image
			blue_mask = cv2.inRange(npimage, WATER_MIN_TERRAIN, WATER_MAX_TERRAIN)
		
			#width, height = subimg.size
			#if subimg.size != (64, 64):
			#	print('error!')

			#water = cv2.countNonZero(blue_mask)
			#water_percent = water / (width*height)
			
			# #print('w: ', water_percent)
			# if water_percent < 0.05 or water_percent > 0.95: #reject tile
			# 	continue
			#
			# if not strictly_coastline(blue_mask, image.size): #reject tile
			# 	continue
			
			grayscale = convert_tile_to_binary(npimage, blue_mask)
			#subfilename = filename.split('\\')[1].rsplit('.', 1)[0] + '_' + str(subidx) + '.png'
			#im.save(os.path.join(directory_out, subfilename), 'PNG')

			grayscale = grayscale[:, :, 0]
			image_data[idx * multiplier + subidx] = grayscale

			#subfilename = filename.rsplit('.', 1)[0] + '_' + str(subidx) + '.png'
			coordinates[idx * multiplier + subidx] = (lat, lon)

		image.close()

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

	hdf5_path = 'coastlines_binary_' + str(size) + '_images.hdf5'
	hdf5_file = h5py.File(hdf5_path, mode='w')
	imgs_shape = (num_images, size, size)  # image_data.shape  #
	hdf5_file.create_dataset("images", imgs_shape, dtype='uint8')
	hdf5_file["images"][...] = image_data

	hdf5_file.create_dataset("coordinates", data=coordinates)

	hdf5_file.close()
	print('...finished creating hdf5')
