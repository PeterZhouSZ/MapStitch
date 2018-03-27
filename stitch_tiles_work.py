# Stitch tiles together to create seamless texture
#@ author: Anna Frühstück


# imports
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import random
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

#######################################################################################################################
# SPOTIFY ANNOY PYTHON LIBRARY (UNUSABLE BECAUSE ONLY RETURNS 10 NN)
#######################################################################################################################
def generate_annoy_hash(images, filename):
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


def get_nearest_edges(engine, edge, num_results=10000, max_distance=100000):
	# adding one to edge to avoid zero vector (which annoy doesn't handle)
	(nn_idxs, nn_dists) = engine.get_nns_by_vector(edge, num_results, include_distances=True)
	#filter results by distances
	nearest_edge_idxs, distances = zip(* [[e, d] for e, d in zip(nn_idxs, nn_dists) if d <= max_distance ])

	return list(nearest_edge_idxs), list(distances)

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


hdf5_path = 'coastlines_binary_128_images.hdf5'  # address of hdf5 data file ##'binary_test_data.hdf5'
hdf5_file = h5py.File(hdf5_path, "r")

# load images from hdf5
images = hdf5_file["images"]

# take a subset: only every 4th image
# images = images[0::4]

num_images = images.shape[0]
print('{} images found...'.format(num_images))

# engine = generate_lsh(images)
# engine = generate_nearpy_hash(images)
engine, identifiers = generate_annoy_hash(images, "annoy_database.ann")

# parameters
max_distance = 2048 #1024#
max_results = 10000
image_size = 128
eps = 3
my_dpi = 140  # screen dpi
failure_threshold = 7

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


def init_start_tile(canvas):
	random_index = int(random.choice(range(num_images)))  # initialize top left randomly
	indices[0, 0] = random_index
	canvas[0, :, :] = images[random_index]


def find_matching_tile(tiles_h, tiles_v, indices):
	for tile in tiles_h.keys():
		if tile not in indices and tile in tiles_v:
			if (tiles_h[tile] + 1) % 4 == tiles_v[tile]:  # check from orientation whether target edges form a corner
				return tile, tiles_v[tile]
	return -1, 0

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
		num_backtrack = np.random.randint(failure_count)  # choose how many steps to backtrack according to the number of failures at this location

		if failure_count > failure_threshold:
			num_backtrack = tiles_w + 1 #if failing too often, track back a full row

		success[current_index - num_backtrack:] = [False] * (num_tiles - current_index + num_backtrack)
		indices[current_index - num_backtrack:] = [0] * (num_tiles - current_index + num_backtrack)

		failure = (current_index, failure_count + 1)
		print('no tiles found: failed {} times, backtracking {} tiles'.format(failure_count, num_backtrack))
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

	while not np.all(success):
		current_index = np.argmin(success) # find first occurrence of False in success

		if current_index == 0: # for start tile, randomly initialize tile
			init_start_tile(canvas)
			success[current_index] = True
			continue

		x = current_index % tiles_w
		y = int(np.floor(current_index / tiles_w))

		print('Position [{}, {}]'.format(x, y))#, end='\r', flush=True)

		possible_tiles_h = []
		possible_tiles_v = []

		random_choice = np.random.rand(1)

		if x != 0:  # if not left border, consider edge with left neighbor
			left_tile_index = indices[ current_index - 1 ] #indices[x - 1, y]
			tile_left_right_edge = get_tile_edge_from_canvas(canvas, image_size, RIGHT, current_index - 1, flip=True)
			left_all_water = np.array_equal(tile_left_right_edge, np.zeros(image_size))
			left_all_land = np.array_equal(tile_left_right_edge, 255*np.ones(image_size))

			idxs_in_list_h, distances_h = get_nearest_edges(engine, tile_left_right_edge ,num_results=max_results,  max_distance=max_distance)
			possible_tiles_h_dict = dict(identifiers[idxs_in_list_h])
		possible_tiles_h = possible_tiles_h_dict.keys()
			# print('# horizontal choices {}'.format(len(possible_tiles_h)))
			#if len(possible_tiles_h) > 1:  # avoid double occurrences if possible
			#			possible_tiles_h = [i for i in possible_tiles_h if i[0] not in indices]

		if y != 0:  # if not top border, consider edge with top neighbor
			top_tile_index = indices[ current_index - tiles_w ] #indices[x, y - 1]
			tile_top_bottom_edge = get_tile_edge_from_canvas(canvas, image_size, BOTTOM, current_index - tiles_w, flip=False)
			top_all_water = np.array_equal(tile_top_bottom_edge, np.zeros(image_size))
			top_all_land = np.array_equal(tile_top_bottom_edge, 255*np.ones(image_size))

			idxs_in_list_v, distances_v = get_nearest_edges(engine, tile_top_bottom_edge, num_results=max_results, max_distance=max_distance)
			possible_tiles_v_dict = dict(identifiers[idxs_in_list_v])
			possible_tiles_v = possible_tiles_v_dict.keys()
			# print('# vertical choices {}'.format(len(possible_tiles_v)))
			#if len(possible_tiles_v) > 1: #avoid double occurrences if possible
			#	possible_tiles_v = [i for i in possible_tiles_v if i[0] not in indices]

		#disallow using the same tile multiple times
		current_used_idxs = indices[np.nonzero(indices)]
		#a = idxs_in_list_h

		#possible_tiles_h, orientations_h = zip(*identifiers[idxs_in_list_h])
		#possible_tiles_v, orientations_v = zip(*identifiers[idxs_in_list_v])

		#possible_tiles_h = [ idx for [idx, orientation] in identifiers[idxs_in_list_h] ]# if idx not in current_used_idxs ]
		#possible_tiles_v = [ idx for [idx, orientation] in identifiers[idxs_in_list_v] ]#if idx not in current_used_idxs ]

		if x == 0 and len(possible_tiles_v) > 0:  # handle first column (disregard left neighbor)
			if top_all_water and random_choice < 0.5:  # 50% chance of all water tile
				select_tile(all_water, current_index, -1, 0, canvas, indices, success)
				continue
			elif top_all_land and random_choice < 0.5:  # 50% chance of all land tile
				select_tile(all_land, current_index, -1, 0, canvas, indices, success)
				continue

			#best_tiles_v = [i for i in possible_tiles_v if i[2] == possible_tiles_v[0][2]]  # select all tiles with same distance
			#random_index = int(random.choice(range(len(best_tiles_v)))) # randomly pick one of selected tiles
			# tile = best_tiles_v[random_index]
			#
			# index = int(tile[0])
			# orientation = int(tile[1])

			best_tiles_v = []
			for idx, tile in enumerate(possible_tiles_v):  # find all eligible tiles that have similar low distance
				if distances_v[idx] > distances_v[0] + eps:
					break
				if tile not in current_used_idxs:
					best_tiles_v.append(tile)

			if len(best_tiles_v) == 0:
				print('ouch')
				#possible_tiles_v, smallest_dist_v = get_nearest_edges(engine, tile_top_bottom_edge, num_results=max_results, max_distance=max_distance)

			random_tile = int(random.choice(range(len(best_tiles_v))))  # randomly pick one of selected tiles
			index = best_tiles_v[random_tile]
			#info = identifiers[index]
			orientation = int(possible_tiles_v_dict[index])

			# write selected tile to canvas
			select_tile(images[index], current_index, index, orientation, canvas, indices, success)
			continue
		elif y == 0 and len(possible_tiles_h) > 0:  # handle first row (disregard top neighbor)
			if left_all_water and random_choice < 0.5:  # 50% chance of all water tile
				select_tile(all_water, current_index, -1, 0, canvas, indices, success)
				continue
			elif left_all_land and random_choice < 0.5:  # 50% chance of all land tile
				select_tile(all_land, current_index, -1, 0, canvas, indices, success)
				continue

			#best_tiles_h = [i for i in possible_tiles_h if i[2] == possible_tiles_h[0][2]]  # select all tiles with same distance
			#random_index = int(random.choice(range(len(best_tiles_h)))) # randomly pick one of selected tiles
			#tile = best_tiles_h[random_index]

			best_tiles_h = []
			for idx, tile in enumerate(possible_tiles_h):  # find all eligible tiles that have similar low distance
				if distances_h[idx] > distances_h[0] + eps:
					break
				if tile not in current_used_idxs:
					best_tiles_h.append(tile)

			if len(best_tiles_h) == 0:
				print('ouch')
				#possible_tiles_h, smallest_dist_h = get_nearest_edges(engine, tile_left_right_edge, num_results=max_results, max_distance=max_distance)

			random_tile = int(random.choice(range(len(best_tiles_h))))  # randomly pick one of selected tiles
			index = best_tiles_h[random_tile]

			orientation = int( ( possible_tiles_h_dict[index] + 1 ) % 4 )

			# write selected tile to canvas
			select_tile(images[index], current_index, index, orientation, canvas, indices, success)
			# indices[current_index] = index
			# put_tile_in_canvas(canvas, images[index], current_index, orientation)
			# success[current_index] = True
			continue
		elif len(possible_tiles_h) == 0 or len(possible_tiles_v) == 0:  # found no choices in one of both directions, do stochastic backtracking
			failure = backtrack(failure, current_index, success, indices)
			continue

		if left_all_water and top_all_water and random_choice < 0.7:  # 70% chance of all water tile
			select_tile(all_water, current_index, -1, 0, canvas, indices, success)
			continue
		elif left_all_land and top_all_land and random_choice < 0.9:  # 90% chance of all land tile
			select_tile(all_land, current_index, -1, 0, canvas, indices, success)
			continue

		#find tile that fits left and top edge
		common_index, orientation = find_matching_tile(possible_tiles_h_dict, possible_tiles_v_dict, indices)

		if common_index > -1: # found tile
			select_tile(images[common_index], current_index, common_index, orientation, canvas, indices, success)
			continue
		else: # do stochastic backtracking
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
	plt.savefig('output/' + filename + '.png', dpi =300, bbox_inches='tight', pad_inches = 0) # my_dpi
	plt.show()


pr = cProfile.Profile()
pr.enable()

tiles_w = 10

# while tiles_w < 31: #LOOooooOOOOoooOP
tiles_h = round(tiles_w * (2/3))

fig = plt.figure(figsize=((image_size * tiles_h) / my_dpi, (image_size * tiles_w) / my_dpi), dpi=my_dpi)
plt.axis('off')
indices = np.zeros((tiles_w, tiles_h))
canvas = generate_map(tiles_w, tiles_h)
save_map(canvas, tiles_w, tiles_h)
# tiles_w += 1
# tiles_h += 1

pr.disable()
# after your program ends
pr.print_stats(sort="cumtime")
