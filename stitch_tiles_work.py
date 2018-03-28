# Stitch tiles together to create seamless texture
#@ author: Anna Frühstück


# imports
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import random
import re
import numpy as np
from lshash.lshash import LSHash
import h5py
from IPython import display
from tqdm import tqdm
from tqdm import trange
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from annoy import AnnoyIndex
import os.path
import time
import cProfile
from collections import OrderedDict

# constants
TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

LT = 0
TR = 1
RB = 2
BL = 3

#######################################################################################################################
# SPOTIFY ANNOY PYTHON LIBRARY (UNUSABLE BECAUSE ONLY RETURNS 10 NN)
#######################################################################################################################
##TODO do simultaneously (stupid!)
def generate_edge_hash(images, filename):
	engine = AnnoyIndex(images.shape[1], metric='euclidean')  # Length of item vector that will be indexed

	full_filename = os.path.join(os.getcwd(), filename)
	if os.path.isfile(full_filename):
		engine.load(full_filename)

		identifiers = np.column_stack((np.floor(np.arange(0, num_images, 0.25)), np.tile(range(4), (1, num_images))[0]))
		return engine, identifiers

	identifiers = np.empty((4 * num_images, 2), dtype=int)
	ct = 0
	for idx, image in enumerate(tqdm(images)):
		# for i in xrange(1000):
		(top, right, bottom, left) = get_all_edges_from_array(image)

		# adding one to each edge to avoid zero vector (which annoy doesn't handle)
		engine.add_item(ct,   top)
		engine.add_item(ct+1, right)
		engine.add_item(ct+2, bottom)
		engine.add_item(ct+3, left)
		identifiers[ct  ] = [idx, 0]
		identifiers[ct+1] = [idx, 1]
		identifiers[ct+2] = [idx, 2]
		identifiers[ct+3] = [idx, 3]
		ct += 4
	engine.build(10)  # 10 trees
	if not os.path.isfile(full_filename):
		engine.save(filename)

	return engine, identifiers

def generate_corner_hash(images, filename):
	engine = AnnoyIndex(2*images.shape[1], metric='euclidean')  # Length of item vector that will be indexed

	full_filename = os.path.join(os.getcwd(), filename)

	if os.path.isfile(full_filename):
		engine.load(full_filename)

		identifiers = np.column_stack((np.floor(np.arange(0, num_images, 0.25)), np.tile(range(4), (1, num_images))[0]))
		return engine, identifiers

	identifiers = np.empty((4 * num_images, 2), dtype=int)
	ct = 0
	for idx, image in enumerate(tqdm(images)):
		# for i in xrange(1000):
		(top, right, bottom, left) = get_all_edges_from_array(image)

		corner_left_top = np.concatenate([left, top])
		corner_top_right = np.concatenate([top, right])
		corner_right_bottom = np.concatenate([right, bottom])
		corner_bottom_left = np.concatenate([bottom, left])
		# adding one to each edge to avoid zero vector (which annoy doesn't handle)
		engine.add_item(ct,   corner_left_top)
		engine.add_item(ct+1, corner_top_right)
		engine.add_item(ct+2, corner_right_bottom)
		engine.add_item(ct+3, corner_bottom_left)
		identifiers[ct  ] = [idx, 0]
		identifiers[ct+1] = [idx, 1]
		identifiers[ct+2] = [idx, 2]
		identifiers[ct+3] = [idx, 3]
		ct += 4
	engine.build(10)  # 10 trees
	if not os.path.isfile(full_filename):
		engine.save(filename)

	return engine, identifiers

def get_nearest_edges(engine, edge, num_results=10000, max_distance=100000):
	# adding one to edge to avoid zero vector (which annoy doesn't handle)
	(nn_idxs, nn_dists) = engine.get_nns_by_vector(edge, num_results, include_distances=True)
	#filter results by distances
	nearest_edge_idxs, distances = zip(* [[e, d] for e, d in zip(nn_idxs, nn_dists) if d <= max_distance ])

	return list(nearest_edge_idxs), list(distances)

def get_nearest_corners(engine, corner, num_results=10000, max_distance=100000):
	# adding one to edge to avoid zero vector (which annoy doesn't handle)
	(nn_idxs, nn_dists) = engine.get_nns_by_vector(corner, num_results, include_distances=True)
	#filter results by distances
	nearest_corner_idxs, distances = zip(* [[e, d] for e, d in zip(nn_idxs, nn_dists) if d <= max_distance ])

	return list(nearest_corner_idxs), list(distances)


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
	canvas = np.zeros((tiles_h * tiles_w, image_size, image_size))
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
	return edge


def get_tile_edges_from_canvas(canvas, tile_size, i):
	top = canvas[i, 0, :]  # TOP
	right = canvas[i, :, tile_size - 1]  # RIGHT
	bottom = np.flip(canvas[i, tile_size - 1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left = np.flip(canvas[i, :, 0], 0)  # LEFT, flip to preserve clockwise order
	return top, right, bottom, left


def init_start_tile(canvas, indices):
	random_index = int(random.choice(range(num_images)))  # initialize top left randomly
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

def get_image_from_canvas(canvas, tiles_w, tiles_h):
	image = np.zeros((image_size * tiles_h, image_size * tiles_w))

	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = int(np.floor(index / tiles_w))
		image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = canvas[index]

	return image

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
			# if np.random.rand(1) > 0.5:
			# 	success.fill(0) #restart
			# 	indices.fill(0)
			# 	print('failed too often, restart')
			# 	return 0, 0
			# else:
			num_backtrack = tiles_w + 1 #track back a full row

		success[current_index - num_backtrack:] = [False] * (num_tiles - current_index + num_backtrack)
		indices[current_index - num_backtrack:] = [0] * (num_tiles - current_index + num_backtrack)

		failure = (current_index, failure_count + 1)
		print('no tiles found at tile #{}: failed {} times, backtracking {} tiles'.format(current_index, failure_count, num_backtrack))
	else:
		failure = (current_index, 1)  # set failure index, but do nothing (try again same tile)
	return failure


def generate_map(tiles_w, tiles_h):
	num_tiles = tiles_w * tiles_h

	canvas = np.zeros((tiles_h * tiles_w, image_size, image_size))
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
			init_start_tile(canvas, indices)
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

		if left_all_water and top_all_water and np.random.rand(1) < 0.9:  # 70% chance of all water tile
			select_tile(all_water, current_index, -1, 0, canvas, indices, success)
			continue
		elif left_all_land and top_all_land and np.random.rand(1) < 0.95:  # 95% chance of all land tile
			select_tile(all_land, current_index, -1, 0, canvas, indices, success)
			continue

		if x == 0:  # handle first column (disregard left neighbor)
			list_idxs_y, distances_y = get_nearest_edges(edge_hash, tile_top_bottom_edge, num_results=max_results, max_distance=max_distance)
			matches_y, orientations = zip(*edge_identifiers[list_idxs_y])
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
				index = random_tile[0]
				orientation = int(random_tile[1])

				# write selected tile to canvas
				select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				continue
		elif y == 0:  # handle first row (disregard top neighbor)

			list_idxs_x, distances_x = get_nearest_edges(edge_hash, tile_left_right_edge, num_results=max_results, max_distance=max_distance)
			matches_x, orientations = zip(*edge_identifiers[list_idxs_x])
			#matches_x_dict = dict(zip(matches_x, zip(orientations, distances_x)))

			best_matches_x = []
			for idx, tile in enumerate(matches_x):  # find all eligible tiles that have similar low distance
				if distances_x[idx] > distances_x[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches_x.append((tile, orientations[idx]))

			num_best = len(best_matches_x)
			if num_best > 0:
				random_tile = best_matches_x[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
				index = random_tile[0]
				orientation = int( (  random_tile[1] + 1 ) % 4 )

				# write selected tile to canvas
				select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				continue
		else:
			corner = np.concatenate([tile_left_right_edge, tile_top_bottom_edge])
			list_idxs, distances = get_nearest_corners(corner_hash, corner, num_results=max_results, max_distance=max_distance)
			matches, orientations = zip(*corner_identifiers[list_idxs])

			best_matches = []
			for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
				if distances[idx] > distances[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches.append((tile, orientations[idx]))

			num_best = len(best_matches)
			if num_best > 0:
				random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
				index = random_tile[0]
				orientation = int(random_tile[1])
				select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				continue
		#find tile that fits left and top edge
		#common_index, orientation = find_matching_tile(matches_x_dict, matches_y_dict, indices)

		# if common_index > -1: # found tile
		# 	select_tile(images[common_index], current_index, common_index, orientation, canvas, indices, success)
		# 	continue
		# else:

		# do stochastic backtracking
		failure = backtrack(failure, current_index, success, indices)
	return canvas


def save_map(canvas, tiles_w, tiles_h):
	image = get_image_from_canvas(canvas, tiles_w, tiles_h)
	timestr = time.strftime("%d_%m_%H%M")
	plt.axis('off')
	plt.imshow(image, origin="upper", cmap="gray")
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.margins(0, 0)
	plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
	plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())

	filename = str(image_size) + 'x' + str(image_size) + '_' + timestr + '_' + str(tiles_w) + 'x' + str(tiles_h)
	plt.savefig('output/' + filename + '.png', dpi=1000, bbox_inches='tight', pad_inches = 0) # my_dpi
	plt.show()


#pr = cProfile.Profile()
#pr.enable()


database_path = 'coastlines_binary_128_images.hdf5'  # address of hdf5 data file ##'binary_test_data.hdf5'
image_size = int(re.search(r'\d+', database_path).group()) #auto-extract integer from string
hdf5_file = h5py.File(database_path , "r")

# load images from hdf5
images = hdf5_file["images"]
num_images = images.shape[0]
print('{} images found...'.format(num_images))

# engine = generate_lsh(images)
# engine = generate_nearpy_hash(images)
edge_hash, edge_identifiers = generate_edge_hash(images, "hash_database_" + str(image_size) + "_edges.ann")
corner_hash, corner_identifiers = generate_corner_hash(images, "hash_database_" + str(image_size) + "_corners.ann")

# parameters
max_distance = 4096 #1024#
max_results = 10000
eps = 15
my_dpi = 140  # screen dpi
failure_threshold = 7

tiles_w = 30
tiles_h = 30
# while tiles_w < 31: #LOOooooOOOOoooOP
#tiles_h = round(tiles_w * (2/3))

fig = plt.figure(figsize=((image_size * tiles_h)/1000, (image_size * tiles_w)/1000), dpi=200)
plt.axis('off')
canvas = generate_map(tiles_w, tiles_h)
save_map(canvas, tiles_w, tiles_h)

#pr.disable()
# after your program ends
#pr.print_stats(sort="cumtime")
