from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np
from view_content_h5py import view_content_of_h5py

def subdivide(image, w_tiles, h_tiles, swidth, sheight):
	out = []
	for sh in range(h_tiles):
		for sw in range(w_tiles):
			subimg = image[sh * sheight:(sh + 1) * sheight, sw * swidth:(sw + 1) * swidth]
			out.append(subim 	g)

	return out

if __name__ == '__main__':
	level = 11
	data_dir = 'D:\\Data\\ESRI\\' + str(level) + '\\'
	power_of_two = 2 ** level
	sizes = [ 64 ]#, 128 ]
	max_elements = 100000
	input_size = 256

	shrink = True
	subdivide = False

	print("processing {} potential tiles at level {}".format(power_of_two*power_of_two, level))

	chunk_size = 1000
	hdf5_files = [None] * len(sizes)
	images_datasets =  [None] * len(sizes)
	coordinates_datasets = [None] * len(sizes)
	chunk_images = [None] * len(sizes)
	chunk_coordinates_shape = (chunk_size, 2)
	chunk_coordinates = np.empty(chunk_coordinates_shape)

	for h, size in enumerate(sizes):
		multiplier = 1
		if subdivide:
			multiplier = (input_size // size) * (input_size // size)
			print("multiply by {} subimages".format(multiplier))
			num_images *= multiplier

		hdf5_path = 'data/esri_eleven_{}x{}_100K.hdf5'.format(size, size)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		hdf5_files[h] = hdf5_file

		chunk_images_shape = (chunk_size, size, size, 3)

		imgs_shape = (0, size, size, 3) #image_data.shape  #
		coords_shape = (0, 2) #image_data.shape  #
		images_dataset = hdf5_file.create_dataset("images", imgs_shape, maxshape=(None, size, size, 3), dtype='uint8', chunks=chunk_images_shape)
		coordinates_dataset = hdf5_file.create_dataset("coordinates", coords_shape, maxshape=(None, 2), dtype='uint32', chunks=chunk_coordinates_shape)

		images_datasets[h] = images_dataset
		coordinates_datasets[h] = coordinates_dataset
		chunk_images[h] = np.empty(chunk_images_shape)

	power_of_two = 2 ** level
	#idx = 0

	# x_range_min = int(power_of_two * 0.649)
	# x_range_max = int(power_of_two * 0.725)
	# y_range_min = int(power_of_two * 0.288)
	# y_range_max = int(power_of_two * 0.346)
	# print("take tiles between x = [{}, {}] and y = [{}, {}]".format(x_range_min, x_range_max, y_range_min, y_range_max))

	indices = np.random.permutation(np.arange(int(power_of_two*power_of_two*0.72))) #
	count = 0
	chunk_count = 0
	for index in tqdm(indices):
		if count > max_elements:
			break
		#index = indices[x]
		y = index % power_of_two
		x = int(index / power_of_two)

		filename_split = str(level) + '_' + str(y) + '_' + str(x) + '.jpg'
		filename = (data_dir + filename_split)

		# if y < y_range_min or y > y_range_max or x < x_range_min or x > x_range_max:
		# 	continue
		try:
			load_image = Image.open(filename)
		except FileNotFoundError:
			continue

		chunk_coordinates[count % chunk_size, :] = (x, y)

		for h, size in enumerate(sizes):
			if shrink:
				image = load_image.resize([size, size], Image.ANTIALIAS)

			image = np.array(image)
			#if len(set(image[:, :, 2].flatten())) < 10:  # too few colors in tile
			#	continue

			chunk_images[h][count % chunk_size, :] = image[None] #store tile to chunk
		if count % chunk_size == 0 and count > 0: #save chunk to file
			for h, size in enumerate(sizes):
				images_datasets[h].resize((count, size, size, 3))
				images_datasets[h][chunk_count:] = chunk_images[h]

				coordinates_datasets[h].resize((count, 2))
				coordinates_datasets[h][chunk_count:] = chunk_coordinates

			chunk_count += chunk_size
		count += 1
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

	if count % chunk_size != 0:
		for h, size in enumerate(sizes):
			last_idx = count % chunk_size
			images_datasets[h].resize((count, size, size, 3))
			images_datasets[h][chunk_count:] = chunk_images[h][:last_idx]

			coordinates_datasets[h].resize((count, 2))
			coordinates_datasets[h][chunk_count:] = chunk_coordinates[:last_idx]

			hdf5_files[h].close()
	print('...finished creating hdf5')

	view_content_of_h5py(hdf5_path, 64)