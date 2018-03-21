#!/usr/bin/python

import glob
from io import BytesIO
from PIL import Image
import os, sys

import time
import random
import cv2
import numpy as np

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
	#filename_list = []
	directory_128 = os.path.join(os.getcwd(), 'coastlines_128') #'coastlines_binary_128')
	
	#create_dir(directory_128)
	
	files = list(glob.iglob('coastlines_256/*.png' ))
	num_images = len(files)
	
	print('{} images found...'.format(num_images))
	printProgressBar(0, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)
	
	for idx, filename in enumerate(files):#glob.glob('coastlines_terrain_14/*.png'): #assuming png format
		image = Image.open(filename)
		
		#print(image.size)
		subimgs = subdivide_into_four(image)
		for subidx, subimg in enumerate(subimgs):
			cimage = subimg.convert("RGB")
			npimage = np.array(cimage)
			#mask blue parts in image
			blue_mask = cv2.inRange(npimage, WATER_MIN_TERRAIN, WATER_MAX_TERRAIN)
		
			width, height = subimg.size
			if(subimg.size != (128, 128)):
				print('error!')
			#print(subimg.size)
			water = cv2.countNonZero(blue_mask)
			water_percent = water / (width*height)
			
			#print('w: ', water_percent)
			if water_percent < 0.05 or water_percent > 0.95: #reject tile
				continue
				
			if not strictly_coastline(blue_mask, image.size): #reject tile
				continue
			
			#im = convert_tile_to_binary(npimage, blue_mask)
			subfilename = filename.split('\\')[1].rsplit('.', 1)[0] + '_' + str(subidx) + '.png'
			subimg.save(os.path.join(directory_128, subfilename), 'PNG')	
		
		image.close()
		printProgressBar(idx + 1, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)