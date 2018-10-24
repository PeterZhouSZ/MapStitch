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

level = 11
data_dir = 'D:\\Data\\ESRI\\' + str(level) + '\\'

files_in_directory = list(glob.glob(data_dir + '*.jpg'))

input_size = max(Image.open(files_in_directory[0]).size)
	
power_of_two = 2 ** level

level_up = 3
upsample_power = 2 ** level_up
print('upsample level: {}'.format(upsample_power))
data_dir_up = 'D:\\Data\\ESRI\\' + str(level + level_up) + '\\'

output_size = power_of_two*input_size

num_images = len(files_in_directory)
print("{} files of size {}x{} found in directory".format(num_images, input_size, input_size))

start_x = int(power_of_two * 0.8222)#0.285)#0
end_x = int(power_of_two * 0.910)#* 0.285)#0.354)#power_of_two
start_y = int(power_of_two * 0.560)#* 0.485)#0
end_y = int(power_of_two * 0.588)#* 0.486)#0.532)# power_of_two

upsampled_output_image = np.zeros((input_size * upsample_power, input_size * upsample_power, 3), dtype='uint8')

#for x in range(start_x, end_x + 1):
#    for y in range(start_y, end_y + 1):
for x in range(start_x, end_x + 1, upsample_power):
    for y in range(start_y, end_y + 1, upsample_power):

        #filename_split = str(level) + '_' + str(y) + '_' + str(x) + '.jpg'
        #filename = (data_dir + filename_split)
        
        #copyfile(filename, 'D:\\Data\\ESRI\\select\\' + filename_split)

        #upsampled_x_start = upsample_power * x
        #upsampled_y_start = upsample_power * y
        
        save = True

        for i in range(upsample_power*upsample_power):
            y_sub = i % upsample_power
            x_sub = int(i / upsample_power)
        #for x_sub in range(upsample_power):
        #    for y_sub in range(upsample_power):
            filename_up_split = str(level) + '_' + str(y + y_sub) + '_' + str(x + x_sub) + '.jpg'
            filename_up = (data_dir + filename_up_split)
            try:    
                image = Image.open(filename_up)    
                upsampled_output_image[y_sub * input_size:(y_sub + 1) * input_size, x_sub * input_size:(x_sub + 1) * input_size] = np.array(image)
            except FileNotFoundError:
                print("File {} not found, skipping this part".format(filename_up) ) 
                save = False
                break
        if save:        
            out_img = Image.fromarray(upsampled_output_image)
            out_img.save('D:\\Data\\ESRI\\select_desert2\\' + str(level) + '_' + str(y) + '_' + str(x) + '_' + str(upsample_power) + 'x.jpg')

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

#out_img.save('results/arcgis_level_'+str(level)+'_x=['+str(start_x)+','+str(end_x)+']_y=['+str(start_y)+','+str(end_y)+']_world.png', "PNG")
