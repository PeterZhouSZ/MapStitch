from PIL import Image as PILImage
from tqdm import tqdm
import colorsys
import math
import numpy as np
import time
from wand.image import Image
from sklearn.cluster import KMeans
from skimage.transform import resize
import matplotlib.pyplot as plt

level = 6
data_dir = 'D:\\Data\\ESRI\\{}\\'.format(level)

save_path = 'D:\\Data\\ESRI\\'

input_size = 256

aspect_ratio = (2, 3) #(height, width)
t_size = 256

power_of_two = 2 ** level

#load and process images
start = time.time()
print('************************************************')
print('load and process files...')
array_items = np.zeros((power_of_two*power_of_two, t_size*t_size*3))

count = 0
location = np.zeros((power_of_two*power_of_two, 2), int)

#pix_size = 2 ** (12 - level)  # t_size
default_color = np.uint8(np.c_[0, 0, 0])  # default color is light blue

#NORTH AFRICA #WORLD CUT TOP BOTTOM #SOUTH A MERICA
start_x = int(power_of_two * 0.474) # 0 # int(power_of_two * 0.285)
end_x   = int(power_of_two * 0.586) # power_of_two # int(power_of_two * 0.354)
start_y = int(power_of_two * 0.410) # int(power_of_two * 0.15) # int(power_of_two * 0.485)
end_y   = int(power_of_two * 0.470) # int(power_of_two * 0.70) # int(power_of_two * 0.532)
range_x = end_x - start_x
range_y = end_y - start_y

output_image = np.tile(default_color, reps=(t_size * range_y, t_size * range_x, 1))
#for i in tqdm(range(power_of_two*power_of_two)):#*power_of_two)):
#    y = int(i % power_of_two)
#    x = int(i / power_of_two)

for y in tqdm(range(0, range_y)):
    for x in range(0, range_x):
        filename_split = str(level) + '_' + str(y + start_y) + '_' + str(x + start_x) + '.jpg'
        filename = (data_dir + filename_split)

        try:
            with Image(filename=filename) as image:
                image.format = 'rgb'  # If rgb image, change this to 'rgb' to get raw values
                image.alpha_channel = False

                image.modulate(brightness=90, saturation=130)
                image.level(0.1, 0.86, gamma=1.3)

                image.resize(t_size, t_size)

                #ycbcrImg = rgb2ycbcr(np.array(image)
                #array_items[count, :] = np.ravel(ycbcrImg)
                output_image[y * t_size:(y + 1) * t_size, x * t_size:(x + 1) * t_size] = np.asarray(bytearray(image.make_blob()), dtype=np.uint8).reshape((t_size, t_size, 3))
                count += 1
        except FileNotFoundError:
            #print("File {} not found, skipping tile".format(filename_split) )
            continue
print('{} files processed'.format(count))
out_img = PILImage.fromarray(output_image)
out_img.save(save_path + 'cutout_desert_level_' + str(level) + '_image_enhanced.jpg')


end = time.time()
print('{:.2f} seconds elapsed'.format(end-start))
