
#!/usr/bin/python

import os

import matplotlib.pyplot as plt
import numpy as np
from os.path import isdir
from PIL import Image
import glob
import h5py
from numpy.random import shuffle
from tqdm import tqdm

WATER_MIN_TERRAIN = np.array([155, 195, 230], dtype=np.uint8) #minimum value of blue pixel in RGB order
WATER_MAX_TERRAIN = np.array([215, 240, 255], dtype=np.uint8) #maximum value of blue pixel in RGB order

def resize_tile(tile, width, height):
	return tile.resize((width, height), Image.ANTIALIAS)  
	
def convert_tile_to_binary(npimage, mask):
	npimage[np.where(blue_mask == [000])] = [255,255,255] 	# blue image parts turn black
	npimage[np.where(blue_mask == [255])] = [000,000,000]   # everything else turns white

	#convert back to numpy image
	return Image.fromarray(npimage)

#remove lake and river tiles	
def strictly_coastline(mask, size):
	width, height = size
	
	edges = [mask[0,:], mask[1:,-1], mask[-1,:-1], mask[1:-1,0]]
	edges_only = np.concatenate(edges)
	water_percent = np.count_nonzero(edges_only) / (2 * (width + height))
	return water_percent > 0.15

def create_dir(name):		
	try:
		os.makedirs(name)
	except OSError as e:
		print('Directory already exists!')
		
if __name__ == "__main__":
	# image_list = []
	# filename_list = []
	# directory_64 = os.path.join(os.getcwd(), 'coastlines_binary_64')
	#
	# create_dir(directory_64)
	#
	# files = list(glob.iglob('coastlines_binary_128/*.png' )) #assuming png format
	# num_images = len(files)
	#
	# print('{} images found...'.format(num_images))
	# printProgressBar(0, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)
	#
	#
	# test_idxs = random.sample(range(20), 10)
	# print(test_idxs)
	#
	# x = np.arange(9).reshape((3, 3))
	# print(x)
	# print(x.transpose())
	# print(x)
	# x1 = np.rot90(x)
	# print(x1)
	# x1t = x1.transpose()
	# print(x1t)
	# print(np.rot90(x1t, 3))
	#
	# print('***')
	# x1 = np.rot90(np.rot90(x).transpose(), 3)
	# print(x1)
	# print(x)
	# x = np.rot90(x)
	# print(x)
	# x = np.rot90(x)
	# print(x)


	#print(np.arange(9).reshape((3, 3)))
	#orsv = [0, 1, 2, 3, 3, 0, 1, 2]
	#orsh = [3, 0, 1, 2, 0, 1, 2, 3]

	# for iter in range(8):
	# 	print("*******************************")
	# 	neighbor_v_orientation = orsv[iter]
	# 	neighbor_h_orientation = orsh[iter]
	#
	# 	tile = np.arange(9).reshape((3, 3))
	#
	# 	print("v orient={}, h orient={}".format(neighbor_v_orientation, neighbor_h_orientation))
	# 	orientation = neighbor_v_orientation
	#
	# 	if (neighbor_v_orientation + 1) % 4 == neighbor_h_orientation:  # target edges form a corner, but tile is mirrored along diagonal
	# 		tile = tile.transpose() # flip along main diagonal
	# 		orientation = neighbor_h_orientation
	#
	# 		if not neighbor_h_orientation % 2 == 0: #flip is actually on the antidiagonal, +2 rotations
	# 			orientation = (orientation + 2) % 4
	#
	# 	#print(tile)
	# 	print("orientation={}".format(orientation))
	# 	final_tile = tile
	# 	if orientation > 0:
	# 		final_tile = np.rot90(final_tile, orientation)
	#
	# 	print(final_tile)


	# possible_neighbors_x_idxs = np.array([8, 7, 10, 2, 19, 8, 7, 10, 3, 11, 5, 13, 19, 4, 14, 3, 15, 13, 5, 4, 14, 15, 9, 1, 20, 1, 9, 21, 2, 20, 12, 21, 11, 12])
	# possible_neighbors_y_idxs = np.array([20, 21, 8, 7, 19, 8, 13, 10, 3, 19, 13, 14, 10, 3, 7, 1, 11, 5, 4, 14, 9, 2, 15, 11, 5, 4, 9, 12, 2, 15, 1, 12, 20, 21, 17, 16, 17, 18, 16, 18])
	#
	# common_neighbors = []
	# for idx, xi in enumerate(possible_neighbors_x_idxs):
	# 	print(xi, idx)
	# 	foo = (possible_neighbors_y_idxs == xi)
	# 	print(foo)
	# 	found_in_y = np.nonzero(possible_neighbors_y_idxs == xi)[0]
	# 	print(found_in_y)
	# 	# yi = np.nonzero(possible_neighbors_y_idxs == xi)[0][0]
	# 	if len(found_in_y > 0):
	# 		yi = found_in_y[0]
	# 		print(yi)
	# 		common_neighbors.append((xi, yi))

	# #x = np.arange(5)#.reshape((4, 3))
	# #y = np.array([3, 4, 5])
	# for xi in np.nditer(x):
	# 	print('xi={}'.format(xi))
	# 	nz = np.nonzero(y == xi)
	# 	#print(nz[0])
	# 	if len(nz[0])>0:
	# 		print(nz[0][0])

	# data_dir = 'coastlines_128/'
	# size = 64
	#
	# if not isdir(data_dir):
	# 	raise Exception("Data directory doesn't exist!")
	#
	# files_in_directory = list(glob.glob(data_dir + '*.png'))
	# num_images = len(files_in_directory)
	# image_data = np.empty([num_images, size, size, 3])
	# idx = 0
	#
	# for filename in tqdm(files_in_directory):  # import all png
	#
	# 	image = np.array(Image.open(filename))#[:, :, 0]
	#
	# 	image_data[idx] = image
	# 	idx += 1
	#
	#
	# shuffle_data = False
	#
	# # to shuffle data
	# if shuffle_data:
	# 	shuffle(image_data)
	#
	#
	# max_val = np.amax(image_data)
	# min_val = np.amin(image_data)
	# print('img range: {} {}'.format(min_val, max_val))
	#
	# hdf5_path = 'coastlines_binary_64_images.hdf5'  # 'coastlines_binary_128_images.hdf5'  # address to where you want to save the hdf5 file
	# hdf5_file = h5py.File(hdf5_path, mode='w')
	# imgs_shape = image_data.shape  # (num_imgs, 64, 64)
	# hdf5_file.create_dataset("images", imgs_shape, dtype='uint8')
	# hdf5_file["images"][...] = image_data
	#
	# hdf5_file.close()
	# print('...finished creating hdf5')
	#


	# num_images = 10
	#
	# a = np.floor(np.arange(0, num_images, 0.25))
	# print(a.shape)
	# print(a)
	# b = np.tile(range(4), (1, num_images))[0]
	# print(b.shape)
	# print(b)
	# c = np.concatenate((a, b), axis=0)
	# print(c)
	# d = np.column_stack((np.floor(np.arange(0, num_images, 0.25)), np.tile(range(4), (1, num_images))[0]))
	# print(d)
	#
	# !/usr/bin/env python

	import cv2
	import numpy as np



	blendMode = cv2.NORMAL_CLONE
	tiles_w = 32
	tiles_h = 16
	image_size = 64
	half_tile = int(image_size / 2)  # 32

	margin = half_tile

	# Read images : src image will be cloned into dst
	#r = 18*np.ones((image_size * tiles_h, image_size * tiles_w), dtype='uint8')
	#g = 32*np.ones((image_size * tiles_h, image_size * tiles_w), dtype='uint8')
	#b = 58*np.ones((image_size * tiles_h, image_size * tiles_w), dtype='uint8')
	#image = np.stack((b, g, r), 2)
	path = 'output/64x64_water_image_05_07_1534_32x16.png' #"output/64x64_water_image_05_07_1855_32x16a.png"
	image = cv2.imread(path)#255*np.ones((image_size * tiles_h, image_size * tiles_w, 3), dtype='uint8')#
	#image = cv2.GaussianBlur(image, (15, 15), 0, 0)
	#cv2.imwrite("output/gaussian-blur-example.jpg", image)
	#image[:,:,:] = image[:,:,:] * [18,32,58]
	obj = cv2.imread(path)
	image = cv2.resize(image, (tiles_w*image_size, tiles_h*image_size))
	cv2.imwrite(path, image)
	obj = cv2.resize(obj, (tiles_w*image_size, tiles_h*image_size))
	margin_inside = 2
	mask = 255 * np.ones((image_size + 2 * margin, image_size + 2 * margin), dtype='uint8')
	mask[margin+margin_inside:-margin-margin_inside, margin+margin_inside:-margin-margin_inside] = 255 * np.zeros((image_size-2*margin_inside, image_size-2*margin_inside), dtype='uint8')

	masked_source = 255 * np.ones((image_size + 2 * margin, image_size + 2 * margin, 3), dtype='uint8')

	print((image.shape))
	print((obj.shape))
	for y in np.arange(1,tiles_h-1):
		for x in np.arange(1, tiles_w-1):
	#for index in range(tiles_w*tiles_h):
	#	x = index % tiles_w
	#	y = int(np.floor(index / tiles_w))
			source = obj[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size]

	#if x == 0 or x == tiles_w-1 or y == 0 or y == tiles_h-1:
			#	image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = source
			#	continue

			masked_source[margin:-margin, margin:-margin] = source
			# beginx = 0
			# endx = 0
			# beginy = 0
			# endy = 0
			# #if x == 0: #do not blend left
			# #	beginx = margin
			# if x < tiles_w-2: #do not blend right
			# 	endx = -margin #do not blend right
			# #if y == 0:
			# #	beginy = margin
			# if y < tiles_h-2:
			# 	endy = -margin #do not blend bottom
			# #mask.fill(255)
			#
			# selection_source = masked_source[beginy:endy, beginx:endx]
			# selection_mask = mask[beginy:endy, beginx:endx]

			if blendMode is not None:
				#poisson blend
				#center = (min(y * image_size + half_tile, max_center_h), min(x * image_size + half_tile, max_center_w))
				#center = (min(x * image_size + half_tile, max_center_w), min(y * image_size + half_tile, max_center_h))
				#center = (x * image_size + half_tile, y * image_size + half_tile)
				center = (x * image_size + half_tile, y * image_size + half_tile)
				#center = (beginx + int((endx - beginx + 1) / 2), beginy + int((endy - beginy + 1) / 2))
				#print(center)
				#masked_source[margin:-margin, margin:-margin] = source
				# masked_source = obj[y * image_size-margin:(y + 1) * image_size+margin, x * image_size-margin:(x + 1) * image_size+margin]
				image = cv2.seamlessClone(masked_source, image, mask, center, blendMode)
			else:
				image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = source
	outpath = path.split('.')[0] + '_gradient.png'
	cv2.imwrite(outpath, image)

	#print(np.tile(range(3), (1, 10)))
	#
	#
	# hdf5_path = 'coastlines_binary_64_images.hdf5'  # address to where you want to save the hdf5 file
	# hdf5_file = h5py.File(hdf5_path, "r")
	# # Total number of samples
	# images = hdf5_file["images"]
	# max_val = np.amax(images)
	# min_val = np.amin(images)
	# print('img range: {} {}'.format(min_val, max_val))
	#
	# idx = np.random.randint(0, images.shape[0], size=16)
	# #idx = range(16)
	# #plt.figure()
	# fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(14, 14), )
	# for ii, ax in zip(idx, axes.flatten()):
	# 	ax.imshow(images[ii], aspect='equal', cmap='gray')  # [:,:,:]
	# 	ax.xaxis.set_visible(False)
	# 	ax.yaxis.set_visible(False)
	# plt.subplots_adjust(wspace=0, hspace=0)
	# plt.show()

	# for idx, filename in enumerate(files):
	# 	image = Image.open(filename)
	# 	image.close()
	# 	printProgressBar(idx + 1, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)