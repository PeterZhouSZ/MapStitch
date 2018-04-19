#!/usr/bin/python

import glob
from io import BytesIO
from PIL import Image
import os, sys

import time
import random
import cv2
import h5py
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import re

#WATER_MIN_TERRAIN = np.array([0, 0, 0], dtype=np.uint8) #minimum value of black pixel in RGB order
#WATER_MAX_TERRAIN = np.array([5, 5, 5], dtype=np.uint8) #maximum value of black pixel in RGB order

WATER_MIN_TERRAIN = np.array([155, 195, 230], dtype=np.uint8) #minimum value of blue pixel in RGB order
WATER_MAX_TERRAIN = np.array([215, 240, 255], dtype=np.uint8) #maximum value of blue pixel in RGB order

def resize_tile(tile, width, height):
	return tile.resize((width, height), Image.ANTIALIAS)  

def subdivide_into_four(image):
	width, height = image.size
	hwidth = round(width / 2)
	hheight = round(height / 2)
	
	box_1 = (0, 0, hwidth, hheight)
	box_2 = (0, hheight, hwidth, height)
	box_3 = (hwidth, 0, width, hheight)
	box_4 = (hwidth, hheight, width, height)
	sub_1 = image.crop(box_1)
	sub_2 = image.crop(box_2)
	sub_3 = image.crop(box_3)
	sub_4 = image.crop(box_4)

	return [sub_1, sub_2, sub_3, sub_4]
	
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
	#print('sc: ', water_percent)
	return water_percent > 0.15

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

	amount_land = []
	num_changes = []

	hdf5_path = 'data/coastlines_binary_128_images.hdf5'  # address of hdf5 data file ##'binary_test_data.hdf5'
	image_size = int(re.search(r'\d+', hdf5_path).group())  # auto-extract integer from string
	hdf5_file = h5py.File(hdf5_path, "r")

	# load images from hdf5
	images = hdf5_file["images"]

	num_images = images.shape[0]

	for idx, image in enumerate(tqdm(images)):#glob.glob('coastlines_terrain_14/*.png'): #assuming png format
		top = image[0, :]  # TOP
		right = image[:, image_size - 1]  # RIGHT
		bottom = np.flip(image[image_size - 1, :], 0)  # BOTTOM, flip to preserve clockwise order
		left = np.flip(image[:, 0], 0)  # LEFT, flip to preserve clockwise order

		top = np.round(top / 255).astype(int) #normalize and round
		right = np.round(right / 255).astype(int)  # normalize and round
		bottom = np.round(bottom / 255).astype(int)  # normalize and round
		left = np.round(left / 255).astype(int)  # normalize and round

		top_amount_land = np.count_nonzero(top)
		right_amount_land = np.count_nonzero(right)
		bottom_amount_land = np.count_nonzero(bottom)
		left_amount_land = np.count_nonzero(left)

		amount_land.extend([top_amount_land, right_amount_land, bottom_amount_land, left_amount_land])

		top_grouped = [x[0] for x in groupby(top)]
		right_grouped = [x[0] for x in groupby(right)]
		bottom_grouped = [x[0] for x in groupby(bottom)]
		left_grouped = [x[0] for x in groupby(left)]

		num_changes.extend([len(top_grouped)-1, len(right_grouped)-1, len(bottom_grouped)-1, len(left_grouped)-1])

	#hist_percentage, bin_edges = np.histogram(percent_land, bins=np.arange(0, 100, 2))
	plt.hist(amount_land, bins=np.arange(1, image_size))  # arguments are passed to np.histogram
	plt.title("Amount of land pixels in edges (without 0)")
	#plt.yscale('log')
	plt.show()

	# hist_percentage, bin_edges = np.histogram(percent_land, bins=np.arange(0, 100, 2))
	plt.hist(num_changes, bins=np.arange(1, max(num_changes)))  # arguments are passed to np.histogram
	#plt.yscale('log')
	plt.title("Number of land/water changes on edges (without 0)")
	plt.show()

	plt.hist(amount_land, bins=np.arange(0, image_size))  # arguments are passed to np.histogram
	plt.title("Amount of land pixels in edges (log scale)")
	plt.yscale('log')
	plt.show()

	#hist_percentage, bin_edges = np.histogram(percent_land, bins=np.arange(0, 100, 2))
	plt.hist(num_changes, bins=np.arange(0, max(num_changes)))  # arguments are passed to np.histogram
	plt.yscale('log')
	plt.title("Number of land/water changes on edges (log scale)")

	plt.show()