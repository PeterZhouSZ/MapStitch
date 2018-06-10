#!/usr/bin/python

import glob
from io import BytesIO
from PIL import Image
import os, sys

import time
import random
import cv2
import numpy as np
import re
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
	directory_128 = os.path.join(os.getcwd(), 'coastlines_128')
	directory_256 = os.path.join(os.getcwd(), 'coastlines_256')
	
	create_dir(directory_128)
	create_dir(directory_256)

	directory = 'coastlines_15'#\\terrain'
	files = list(glob.iglob(directory+'/*.png' ))
	num_images = len(files)
	
	print('{} images found...'.format(num_images))
	printProgressBar(0, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)
	
	for idx, filename in enumerate(files):#glob.glob('coastlines_terrain_14/*.png'): #assuming png format
		m = re.match(r'coastlines_15\\15_(.*)_(.*).png', filename)
		coords = str(m.group(1))
		type = str(m.group(2))
		new_filename = "coastlines_15\\"+type+"\\15_"+coords+".png"
		os.rename(filename, new_filename)

		filename = new_filename
		image = Image.open(filename)
		#print(image.mode)

		image = image.convert('RGB')
		# if type == 'terrain':
		# 	#convert to numpy array for cv2
		# 	npimage = np.array(image)
		# 	#mask blue parts in image
		# 	blue_mask = cv2.inRange(npimage, WATER_MIN_TERRAIN, WATER_MAX_TERRAIN)
		# 	image = convert_tile_to_binary(npimage, blue_mask)

		image = resize_tile(image, 512, 512)
		image.save(filename)#os.path.join(directory+"/"+type, filename.split('\\')[-1]), 'PNG')

		image.close()
		printProgressBar(idx + 1, num_images, prefix = 'Progress:', suffix = 'complete', length = 50)