# Stitch tiles together to create seamless texture
#@ author: Anna Frühstück


# imports
import glob
from PIL import Image, ImageDraw
import PIL.ImageOps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import re
import numpy as np
from lshash.lshash import LSHash
import h5py
from IPython import display
from tqdm import tqdm
from tqdm import trange
#from nearpy import Engine
#from nearpy.hashes import RandomBinaryProjections
import os.path
import time
import cProfile
from lonlat2color import lonlat2rgba
#from evaluate_nearest_neighbor_annoy import generate_hashes, generate_tile_hash, get_nearest_tiles, get_nearest_corners, get_nearest_edges
from evaluate_nearest_neighbor_hnsw import generate_hashes, get_nearest

from matplotlib.collections import PatchCollection
from collections import OrderedDict
import cv2

# constants
TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

LT = 0
TR = 1
RB = 2
BL = 3

#random.seed(42)

#######################################################################################################################
# return a particular edge from image array
def get_edge_from_array(image, edge_type, flip=False):
	edge = None
	if edge_type == TOP:
		edge = image[0, :]  # TOP
	elif edge_type == RIGHT:
		edge =  image[:, image.shape[0]-1]  # RIGHT
	elif edge_type == BOTTOM:
		edge = image[image.shape[1]-1, :]  # BOTTOM
	elif edge_type == LEFT:
		edge = image[:, 0]  # LEFT
	if flip:
		edge = np.flip(edge, 0)
	return edge


# return all edges from image array
def get_all_edges_from_array(image):
	top    = image[0, :]  # TOP
	right  = image[:, image.shape[0]-1]  # RIGHT
	bottom = np.flip(image[image.shape[1]-1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(image[:, 0], 0)  # LEFT, flip to preserve clockwise order
	return top, right, bottom, left

#######################################################################################################################
# helper functions for map generation
#######################################################################################################################

def init_canvas():
	if channels > 1:
		canvas = np.zeros((tiles_h * tiles_w, image_size, image_size, channels), dtype='uint8')
	else:
		canvas = np.zeros((tiles_h * tiles_w, image_size, image_size), dtype='uint8')
	clear_canvas(canvas)
	return canvas


def clear_canvas(canvas):
	canvas.fill(0)


def clear_after(canvas, index):
	empty = np.zeros((image_size, image_size))
	for i in np.arange(index, canvas.shape[0]):
		canvas[i, :, :] = empty


def get_tile_edge_from_canvas(canvas, tile_size, edge_type, i, flip=False):
	edge = None
	if edge_type == TOP:
		edge = canvas[i, 0, :]  # TOP
	elif edge_type == RIGHT:
		edge = canvas[i, :, tile_size - 1]  # RIGHT
	elif edge_type == BOTTOM:
		edge = canvas[i, tile_size - 1, :]  # BOTTOM
	elif edge_type == LEFT:
		edge = canvas[i, :, 0]  # LEFT
	if flip:
		edge = np.flip(edge, 0)
	return edge.flatten('F')


def get_tile_edges_from_canvas(canvas, tile_size, i):
	top = canvas[i, 0, :]  # TOP
	right = canvas[i, :, tile_size - 1]  # RIGHT
	bottom = np.flip(canvas[i, tile_size - 1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left = np.flip(canvas[i, :, 0], 0)  # LEFT, flip to preserve clockwise order
	return top, right, bottom, left


def init_start_tile(canvas, images, indices):
	random_index = int(random.choice(range(images.shape[0])))  # initialize top left randomly
	indices[0] = random_index
	canvas[0, :, :] = images[random_index]


def find_matching_tile(matches_x_dict, matches_y_dict, indices):
	matches = matches_x_dict.keys() & matches_y_dict.keys()

	# # exclude tiles that are already in canvas
	# matches = [ m for m in matches if m not in indices ]
	#
	# orientations_x, distances_x = zip(* [ matches_x_dict[m] for m in matches ])
	# orientations_y, distances_y = zip(* [ matches_y_dict[m] for m in matches ])
	#
	# corners = (orientations_x + 1) % 4 == orientations_y
	# if not any(corners):
	# 	return -1, 0
	#
	# sum_distances = distances_x + distances_y
	# sum_distances = [ s if corner else max(sum_distances) for s, corner in zip(sum_distances, corners) ]
	# #find order in sum_distances
	# order = sorted(range(len(matches)), key=lambda k: sum_distances[k])
	#
	# #pick random from smallest 5 matches
	# pick = order[np.random.randint(5)]
	# return matches[pick], orientations_y[pick]

	sum_distance = 2 * max_distance
	best_match = -1
	best_orientation = 0
	for tile in matches:
		[ orientation_x, distance_x ] = matches_x_dict[tile]
		[ orientation_y, distance_y ] = matches_y_dict[tile]
		if tile in indices or not (orientation_x + 1) % 4 == orientation_y: # check from orientation whether target edges form a corner
			continue
		if distance_x + distance_y < sum_distance: #+ np.random.rand(1) * eps
			sum_distance = distance_x + distance_y
			best_match = tile
			best_orientation = orientation_y
	return best_match, best_orientation


	# for tile in matches_x_dict.keys():
	# 	if tile not in indices and tile in matches_y_dict:
	# 		if (matches_x_dict[tile] + 1) % 4 == matches_y_dict[tile]:  # check from orientation whether target edges form a corner
	# 			return tile, matches_y_dict[tile]
	# return -1, 0

def get_image_from_canvas(canvas, tiles_w, tiles_h, blendMode=None):
	#channels = 1
	if len(canvas.shape) > 3:
		image = np.zeros((image_size * tiles_h, image_size * tiles_w, 3), dtype='uint8')
	else:
		image = np.zeros((image_size * tiles_h, image_size * tiles_w), dtype='uint8')

	mask = 255 * np.ones((image_size, image_size), dtype='uint8')

	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = int(np.floor(index / tiles_w))

		if blendMode is not None:
			#poisson blend
			#mask = np.zeros((image_size * tiles_h, image_size * tiles_w), dtype='uint8')
			#mask[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = 255 * np.ones((image_size, image_size), dtype='uint8')
			source = canvas[index]
			center = (y * image_size + int(image_size / 2)-1, x * image_size + int(image_size / 2)-1)
			image = cv2.seamlessClone(source, image, mask, center, blendMode)
			# normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
			# mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
		else:
			image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = canvas[index]

	return image

def get_annotations(tiles_w, image_size, indices, coordinates):
	patches = []
	colors = []
	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = int(np.floor(index / tiles_w))

		tile = int(indices[index])
		if tile == -1:
			continue
		else:
			(lat, lon) = coordinates[tile]
			if np.isnan(lat) or np.isnan(lon):
				continue
			fc = lonlat2rgba(lon, lat)
			rect = mpatches.Rectangle((x * image_size, y * image_size), image_size, image_size)
			patches.append(rect)
			colors.append(fc)
			#image[((y + 1) * image_size, x * image_size):(x + 1) * image_size] = canvas[index]
	#colors = 100 * np.random.rand(len(patches))
	collection = PatchCollection(patches, alpha=0.35)
	collection.set_facecolors(colors)
	collection.set_edgecolors(colors)
	collection.set_linewidth(0.1)
	#collection.set_array(np.array(colors))
	return collection


def select_tile(tile, list_index, tile_index, orientation, canvas, indices, success):
	indices[list_index] = tile_index
	success[list_index] = True

	if orientation > 0:
		tile = np.rot90(tile, orientation)
	canvas[list_index, :, :] = tile


def backtrack(failure, current_index, success, indices):
	if failure[0] == current_index:  # there was a previous failure at this location
		num_tiles = len(success)
		failure_count = failure[1]
		num_backtrack = failure_count#np.random.randint(failure_count)  # choose how many steps to backtrack according to the number of failures at this location

		if failure_count > failure_threshold: #if failing too often
			if np.random.rand(1) > 0.8:
				success.fill(0) #restart
				indices.fill(0)
				print('failed too often, restart')
				return 0, 0
			else:
				num_backtrack = tiles_w + 1 #track back a full row

		success[current_index - num_backtrack:] = [False] * (num_tiles - current_index + num_backtrack)
		indices[current_index - num_backtrack:] = [0] * (num_tiles - current_index + num_backtrack)

		failure = (current_index, failure_count + 1)
		print('no tiles found at tile #{}: failed {} times, backtracking {} tiles'.format(current_index, failure_count, num_backtrack))
	else:
		failure = (current_index, 1)  # set failure index, but do nothing (try again same tile)
	return failure


def generate_map(tiles_w, tiles_h, images):
	num_tiles = tiles_w * tiles_h
	num_images = images.shape[0]

	channels = 1
	if len(images.shape) > 3:
		channels = images.shape[3]
		canvas = np.zeros((tiles_h * tiles_w, image_size, image_size, channels), dtype='uint8')
	else:
		canvas = np.zeros((tiles_h * tiles_w, image_size, image_size), dtype='uint8')

	indices = np.zeros(num_tiles)
	success = np.zeros(num_tiles, dtype=bool)

	all_water = np.zeros((image_size, image_size))
	all_land = 255 * np.ones((image_size, image_size))

	failure = (0, 0)
	progress_bar = tqdm(total=num_tiles)
	progress = 0

	while not np.all(success):
		current_index = np.argmin(success) # find first occurrence of False in success

		if current_index > progress:
			progress_bar.update(1)
			progress += 1

		if current_index == 0: # for start tile, randomly initialize tile
			init_start_tile(canvas, images, indices)
			success[current_index] = True
			continue

		x = current_index % tiles_w
		y = int(np.floor(current_index / tiles_w))

		current_used_idxs = indices[np.nonzero(indices)]

		left_all_water = True
		left_all_land = True
		top_all_water = True
		top_all_land = True

		if x != 0:  # if not left border, consider edge with left neighbor
			tile_left_right_edge = get_tile_edge_from_canvas(canvas, image_size, RIGHT, current_index - 1, flip=True)
			left_all_water = np.array_equal(tile_left_right_edge, np.zeros(image_size))
			left_all_land = np.array_equal(tile_left_right_edge, 255*np.ones(image_size))

		if y != 0:  # if not top border, consider edge with top neighbor
			tile_top_bottom_edge = get_tile_edge_from_canvas(canvas, image_size, BOTTOM, current_index - tiles_w, flip=False)
			top_all_water = np.array_equal(tile_top_bottom_edge, np.zeros(image_size))
			top_all_land = np.array_equal(tile_top_bottom_edge, 255*np.ones(image_size))

		if x == 0:  # handle first column (disregard left neighbor)
			list_idxs_y, distances_y = get_nearest(edge_hash, tile_top_bottom_edge, num_results=max_results, max_distance=max_distance)
			matches_y, orientations = zip(*identifiers[list_idxs_y])
			#matches_y_dict = dict(zip(matches_y, zip(orientations, distances_y)))

			best_matches_y = []
			for idx, tile in enumerate(matches_y):  # find all eligible tiles that have similar low distance
				if distances_y[idx] > distances_y[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches_y.append((tile, orientations[idx]))

			num_best = len(best_matches_y)
			if num_best > 0:
				random_tile = best_matches_y[int(random.choice(range(num_best)))] # randomly pick one of selected tiles
				index = int(random_tile[0])
				orientation = int(random_tile[1])

				if top_all_water and random.random() > 0.125:
					select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
				elif top_all_land and random.random() > 0.125:
					select_tile(all_land, current_index, num_images+1, 0, canvas, indices, success)
				else:
					# write selected tile to canvas
					select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				continue
		elif y == 0:  # handle first row (disregard top neighbor)

			list_idxs_x, distances_x = get_nearest(edge_hash, tile_left_right_edge, num_results=max_results, max_distance=max_distance)
			matches_x, orientations = zip(*identifiers[list_idxs_x])
			#print(distances_x.shape)
			best_matches_x = []
			for idx, tile in enumerate(matches_x):  # find all eligible tiles that have similar low distance
				#print(distances_x)
				#print(distances_x[0])
				if distances_x[idx] > distances_x[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches_x.append((tile, orientations[idx]))

			num_best = len(best_matches_x)
			if num_best > 0:
				random_tile = best_matches_x[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
				index = int(random_tile[0])
				orientation = int( (  random_tile[1] + 1 ) % 4 )

				if left_all_water and random.random() > 0.125:
					select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
				elif left_all_land and random.random() > 0.125:
					select_tile(all_land, current_index, num_images+1, 0, canvas, indices, success)
				else:
					# write selected tile to canvas
					select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				continue
		else:
			corner = np.concatenate([tile_left_right_edge, tile_top_bottom_edge])
			list_idxs, distances = get_nearest(corner_hash, corner, num_results=max_results, max_distance=max_distance)
			if len(list_idxs) > 0:
				matches, orientations = zip(*identifiers[list_idxs])

				best_matches = []
				for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
					if distances[idx] > distances[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches.append((tile, orientations[idx]))

				num_best = len(best_matches)
				if num_best > 0:
					random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int(random_tile[1])

					if left_all_water and top_all_water and random.random() > 0.075:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and top_all_land and random.random() > 0.075:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					continue

		# do stochastic backtracking
		failure = backtrack(failure, current_index, success, indices)
	return canvas, indices

def fill_hole(npimage, pos):
	top    = npimage[pos[1], pos[0]:pos[0]+image_size]  # TOP
	right  = npimage[pos[1]:pos[1]+image_size, pos[0]+image_size]  # RIGHT
	bottom = np.flip(npimage[pos[1]+image_size, pos[0]:pos[0]+image_size], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(npimage[pos[1]:pos[1]+image_size, pos[0]], 0)  # LEFT, flip to preserve clockwise order

	edges = np.concatenate([top, right, bottom, left])
	list_idxs, distances = get_nearest_tiles(tile_hash, edges, num_results=max_results, max_distance=max_distance)
	matches, orientations = zip(*identifiers[list_idxs])

	best_matches = []
	for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
		if distances[idx] > distances[0] + eps:
			break

		best_matches.append((tile, orientations[idx]))

	num_best = len(best_matches)
	if num_best > 0:
		random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
		tile = images[int(random_tile[0])]
		orientation = int(random_tile[1])
		if orientation > 0:
			tile = np.rot90(tile, orientation)
		npimage[pos[1]:pos[1] + image_size, pos[0]:pos[0] + image_size] = tile

	return npimage

def fill_hole_satellite(original, binary_image, satellite_image, pos):

	top    = original[pos[1], pos[0]:pos[0]+image_size]  # TOP
	right  = original[pos[1]:pos[1]+image_size, pos[0]+image_size-1]  # RIGHT
	bottom = np.flip(original[pos[1]+image_size-1, pos[0]:pos[0]+image_size], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(original[pos[1]:pos[1]+image_size, pos[0]], 0)  # LEFT, flip to preserve clockwise order

	edges = np.concatenate([top, right, bottom, left])
	list_idxs, distances = get_nearest(tile_hash, edges, num_results=max_results, max_distance=max_distance)
	matches, orientations = zip(*identifiers[list_idxs])

	best_matches = []
	for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
		if distances[idx] > distances[0] + eps:
			break

		best_matches.append((tile, orientations[idx]))

	num_best = len(best_matches)
	if num_best > 0:
		random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
		satellite_tile = sat_images[int(random_tile[0])]
		binary_tile = images[int(random_tile[0])]
		orientation = int(random_tile[1])

		if orientation > 0:
			satellite_tile = np.rot90(satellite_tile, orientation)
			binary_tile = np.rot90(binary_tile, orientation)

		#print(tile.shape)

		binary_image[pos[1]:pos[1] + image_size, pos[0]:pos[0] + image_size] = binary_tile
		satellite_image[pos[1]:pos[1] + image_size, pos[0]:pos[0] + image_size] = satellite_tile

	return binary_image, satellite_image


def save_map(image, prefix='', pc=None):
	timestr = time.strftime("%m_%d_%H%M")
	filename = str(image_size) + 'x' + str(image_size) + '_' + prefix + '_' + timestr + '_' + str(tiles_w) + 'x' + str(tiles_h)

	plt.axis('off')
	plt.imshow(image, origin="upper")
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.margins(0, 0)
	plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
	plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())

	plt.savefig('output/' + filename + '.png', dpi=1000, bbox_inches='tight', pad_inches = 0) # my_dpi
	if pc is not None:
		plt.gca().add_collection(pc)
		plt.savefig('output/' + filename + '_annotated.png', dpi=1000, bbox_inches='tight', pad_inches=0)  # my_dpi
	plt.show()

#pr = cProfile.Profile()
#pr.enable()

# parameters
max_distance = 10000 #1024#
max_results = 10000
eps = 2500
failure_threshold = 15

#######################################################################################################################
## hole filling
#
# databases_path = ['data/coastlines_binary_satellite_128_images.hdf5']#['data/coastlines_binary_cleaned_256_images.hdf5', 'data/coastlines_binary_cleaned_128_images.hdf5', 'data/coastlines_binary_cleaned_64_images.hdf5', 'data/coastlines_binary_cleaned_32_images.hdf5']  # address of hdf5 data file ##'binary_test_data.hdf5'
#
# filename = 'exemplars\\Hong_Kong.png'
# original_img = Image.open(filename).convert("RGB")
# oWidth, oHeight = original_img.size
#
# image = np.array(original_img)
# image = image[:, :, 0]
#
# for database_path in databases_path:
# 	image_size = int(re.search(r'\d+', database_path).group()) #auto-extract integer from string
# 	hdf5_file = h5py.File(database_path , "r")
#
# 	# load images from hdf5
# 	images = np.array(hdf5_file["images"])
# 	sat_images = np.array(hdf5_file["satellite"])
# 	coordinates = np.array(hdf5_file["coordinates"])
# 	num_images = images.shape[0]
#
# 	print('{} images found...'.format(num_images))
#
# 	all_water = np.zeros((image_size, image_size))
# 	all_land = 255 * np.ones((image_size, image_size))
#
# 	#images = np.concatenate([images, np.stack((all_water, all_land))])
# 	#coordinates = np.concatenate([coordinates, np.stack(((np.nan, np.nan), (np.nan, np.nan)))])
#
# 	#tile_hash, identifiers = generate_tile_hash(images, "data/hash_database_" + str(image_size) + "_tiles.ann")
# 	tile_hash, identifiers = generate_tile_hash(images, "data/hash_database_satellite_" + str(image_size) + "_tiles.ann")
#
# 	#data_dir = 'coastlines_input_samples/'
# 	#filenames = list(glob.glob(data_dir + '*.png'))
# 	# for filename in filenames:
#
#
# 	#ticks = [32, 192, 352]
# 	ticks_x = np.arange(0, int(np.floor(oWidth/image_size) * image_size), image_size)
# 	ticks_y = np.arange(0, int(np.floor(oHeight/image_size) * image_size), image_size)
# 	#ticks_y = [0, 127, 255, 383]
#
# 	satellite_image = np.zeros((oHeight, oWidth, 3), dtype=np.uint8)
# 	binary_image = np.zeros((oHeight, oWidth), dtype=np.uint8)
# 	for x in ticks_x:
# 		for y in ticks_y:
# 			pos = (x, y)
# 			draw = ImageDraw.Draw(original_img)
# 			draw.rectangle((pos, (pos[0]+image_size, pos[1]+image_size)), outline="red")
# 			#original_img.show()
#
# 			tile = image[pos[1]:pos[1]+image_size, pos[0]:pos[0]+image_size]
# 			#if np.array_equal(tile, all_water) or np.array_equal(tile, all_land): #don't replace those
# 			#	continue
#
# 			#image = fill_hole(image, pos)
# 			new_image = fill_hole_satellite(image, binary_image, satellite_image, pos)
#
# #print(new_image.shape)
# #new_image = np.stack((new_image, new_image, np.zeros((oHeight, oWidth), dtype=np.uint8)), 2)
# #print(new_image.shape)
# new_im = Image.new('RGB', (3*oWidth, oHeight))
#
# new_im.paste(original_img, (0, 0))
# new_im.paste(Image.fromarray(binary_image), (oWidth, 0))
# new_im.paste(Image.fromarray(satellite_image), (2*oWidth, 0))
#
# #new_im.show()
#
# timestr = time.strftime("%m_%d_%H%M")
#
# filename = filename.split('\\')[1]
# filename = filename.split('.')[0]
# #m = re.match(r'coastlines_input_samples\\sample_(.*).png', filename)
# #m = re.match(r'coastlines_terrain_binary/14_(.*),(.*)_terrain.png', filename)
# #num = str(float(m.group(1)))
# #lat = str(float(m.group(1)))
# #lon = str(float(m.group(2)))
# #new_im.save("results/hole_fill_"+lat+","+lon+"_"+timestr+".png", "PNG")
# new_im.save("results/hole_fill_"+filename+"_"+timestr+".png", "PNG")

#######################################################################################################################
## map generating

target_width = 2048
target_height = 768

database_path = 'data/esri_128_images.hdf5' #coastlines_binary_cleaned_128_images.hdf5'
image_size = int(re.search(r'\d+', database_path).group()) #auto-extract integer from string
hdf5_file = h5py.File(database_path , "r")

images = np.array(hdf5_file["images"])

tiles_w = int(np.floor(target_width / image_size))#30
tiles_h = int(np.floor(target_height / image_size))#20

prefix = "hash_esri_" + str(image_size) #"hash_database"
#corner_hash, edge_hash, identifiers = generate_hashes(images, "data/" + prefix + "_" + str(image_size) + "_corners.ann", "data/" + prefix + "_" + str(image_size) + "_edges.ann")
corner_hash, edge_hash, identifiers = generate_hashes(images, "data/" + prefix + "_corners.bin", "data/" + prefix + "_edges.bin")
binary_canvas, indices = generate_map(tiles_w, tiles_h, images)

binary_image = get_image_from_canvas(binary_canvas, tiles_w, tiles_h, None)
save_map(binary_image, 'binary')
# 
# database_path = 'data/satellite_zoom_64_images.hdf5'
# image_size = int(re.search(r'\d+', database_path).group()) #auto-extract integer from string
# hdf5_file = h5py.File(database_path , "r")
# 
# # load images from hdf5
# land_images = np.array(hdf5_file["land"])
# water_images = np.array(hdf5_file["water"])
# 
# tiles_w = int(np.floor(target_width / image_size))#30
# tiles_h = int(np.floor(target_height / image_size))#20
# #tiles_h = round(tiles_w * (2/3))
# 
# 
# # mask_inpaint = np.zeros((tiles_h * image_size, tiles_w * image_size), dtype='uint8')
# # inpaint_w = 10
# # vert_stripe = 255 * np.ones((tiles_h * image_size, 2 * inpaint_w))
# # hor_stripe = 255 * np.ones((2 * inpaint_w, tiles_w * image_size))
# # for t in range(tiles_w-1):
# # 	mask_inpaint[:, (t + 1) * image_size-inpaint_w:(t + 1) * image_size+inpaint_w] = vert_stripe
# #
# # for t in range(tiles_h-1):
# # 	mask_inpaint[(t + 1) * image_size-inpaint_w:(t + 1) * image_size+inpaint_w, :] = hor_stripe
# #
# # save_map(mask_inpaint, 'mask_inpaint')
# 
# 
# fig = plt.figure(figsize=((image_size * tiles_h)/1000, (image_size * tiles_w)/1000), dpi=200)
# plt.axis('off')
# 
# corner_hash, edge_hash, identifiers = generate_hashes(land_images, "data/hash_land_" + str(image_size) + "_corners.ann", "data/hash_land_" + str(image_size) + "_edges.ann")
# land_canvas, indices = generate_map(tiles_w, tiles_h, land_images)
# 
# corner_hash, edge_hash, identifiers = generate_hashes(water_images, "data/hash_water_" + str(image_size) + "_corners.ann", "data/hash_water_" + str(image_size) + "_edges.ann")
# water_canvas, indices = generate_map(tiles_w, tiles_h, water_images)
# 
# # normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
# land_image = get_image_from_canvas(land_canvas, tiles_w, tiles_h, cv2.NORMAL_CLONE)
# water_image = get_image_from_canvas(water_canvas, tiles_w, tiles_h, cv2.NORMAL_CLONE)
# 
# 
# save_map(land_image, 'land_image')
# save_map(water_image, 'water_image')
# #mask land and water images
# land_image = cv2.bitwise_or(land_image, land_image, mask=binary_image)
# 
# binary_image = cv2.bitwise_not(binary_image)
# water_image = cv2.bitwise_or(water_image, water_image, mask=binary_image)
# 
# #pc = get_annotations(tiles_w, image_size, indices, coordinates)
# #combine land and water images
# final = cv2.bitwise_or(land_image, water_image)
# save_map(final, 'satellite')
######################################################################################################################
# pr.disable()
# #after your program ends
# pr.print_stats(sort="cumtime")