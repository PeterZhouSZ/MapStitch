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

level = 9
data_dir = 'D:\\Data\\ESRI\\' + str(level) + '\\'

files_in_directory = list(glob.glob(data_dir + '*.jpg'))

input_size = max(Image.open(files_in_directory[0]).size)
	
power_of_two = 2 ** level

output_size = power_of_two*input_size

num_images = len(files_in_directory)
print("{} files of size {}x{} found in directory".format(num_images, input_size, input_size))

start_x = int(power_of_two * 0.285)#0
end_x = int(power_of_two * 0.354)#power_of_two
start_y = int(power_of_two * 0.485)#0
end_y = int(power_of_two * 0.532)# power_of_two

#full_output_image = np.zeros((input_size * (end_y - start_y + 1), input_size * (end_x - start_x + 1), 3), dtype='uint8')

for filename in tqdm(files_in_directory):  # import all png

	filename_split = filename.split('\\')[-1]
	m = re.match(r'(.*)_(.*)_(.*).jpg', filename_split)
	y = int(m.group(2))
	x = int(m.group(3))

	if x < start_x or x > end_x or y < start_y or y > end_y:
		continue
	#print("image at [{}, {}]".format(x, y))
	x -= start_x
	y -= start_y
	
	copyfile(filename, 'D:\\Data\\ESRI\\select\\' + filename_split)
	#image = Image.open(filename)#[:, :, 0]
	#full_output_image[y * input_size:(y + 1) * input_size, x * input_size:(x + 1) * input_size] = np.array(image)

	
#out_img = Image.fromarray(full_output_image)
# draw = ImageDraw.Draw(out_img)

# color = 'hsl(%d, %d%%, %d%%)' % (25, 100, 50)
# grid_level = 11
# grid_power_of_two = 2 ** grid_level
# grid_size = output_size / grid_power_of_two

# #south american rainforest
# x_range = range(int(grid_power_of_two * 0.286), int(grid_power_of_two * 0.353))
# y_range = range(int(grid_power_of_two * 0.486), int(grid_power_of_two * 0.528))

# for x in x_range: 	
# 	for y in y_range: 
# 		draw.line( ( (x+1) * grid_size, y * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)
# 		draw.line( ( x * grid_size, (y+1) * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)


# color = 'hsl(%d, %d%%, %d%%)' % (50, 100, 50)

# #north african desert
# x_range = range(int(grid_power_of_two * 0.474), int(grid_power_of_two * 0.586))
# y_range = range(int(grid_power_of_two * 0.410), int(grid_power_of_two * 0.453))

# for x in x_range: 	
# 	for y in y_range: 
# 		draw.line( ( (x+1) * grid_size, y * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)
# 		draw.line( ( x * grid_size, (y+1) * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)

# color = 'hsl(%d, %d%%, %d%%)' % (75, 100, 50)

# #russian countryside
# x_range = range(int(grid_power_of_two * 0.649), int(grid_power_of_two * 0.725))
# y_range = range(int(grid_power_of_two * 0.288), int(grid_power_of_two * 0.346))

# for x in x_range: 	
# 	for y in y_range: 
# 		draw.line( ( (x+1) * grid_size, y * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)
# 		draw.line( ( x * grid_size, (y+1) * grid_size, (x+1) * grid_size, (y+1) * grid_size), fill=color)
#out_img.save('results/arcgis_level_'+str(level)+'grid_'+str(grid_level)+'_world.png', "PNG")

out_img.save('results/arcgis_level_'+str(level)+'_x=['+str(start_x)+','+str(end_x)+']_y=['+str(start_y)+','+str(end_y)+']_world.png', "PNG")
