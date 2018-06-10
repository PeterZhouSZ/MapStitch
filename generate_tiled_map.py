# Stitch tiles together to create seamless texture
#@ author: Anna Frühstück


# imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import re
import numpy as np
import h5py
from tqdm import tqdm
import time
from lonlat2color import lonlat2rgba
from evaluate_nearest_neighbor_hnsw import generate_hashes, get_nearest
#from tile_ANN import generate_hashes, generate_tile_hash, get_nearest

from matplotlib.collections import PatchCollection
import cv2

# constants
TOP = 0
RIGHT = 1#
BOTTOM = 2
LEFT = 3

LT = 0
TR = 1
RB = 2
BL = 3

#random.seed(42)

#######################################################################################################################
# helper functions for map generation
#######################################################################################################################

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


def init_start_tile(canvas, images, indices):
	good_pick = False
	while not good_pick:
		random_index = int(random.choice(range(images.shape[0])))  # initialize top left randomly
		if len(set(images[random_index, :, :, 2].flatten())) < 50: 
			continue
		indices[0] = random_index
		canvas[0, :, :] = images[random_index]
		good_pick = True

def get_image_from_canvas(canvas, image_size, tiles_w, tiles_h):
	#channels = 1
	if len(canvas.shape) > 3:
		image = np.zeros((image_size * tiles_h, image_size * tiles_w, 3), dtype='uint8')
	else:
		image = np.zeros((image_size * tiles_h, image_size * tiles_w), dtype='uint8')

	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = index // tiles_w
		image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = canvas[index]

	return image

def get_annotations(tiles_w, image_size, indices, coordinates):
	patches = []
	colors = []
	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = index // tiles_w

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
	collection = PatchCollection(patches, alpha=0.35)
	collection.set_facecolors(colors)
	collection.set_edgecolors(colors)
	collection.set_linewidth(0.1)
	return collection

def get_errormap(tiles_w, image_size, errors):
	patches = []
	colors = []
	max_error = max(errors)
	for index in range(errors.shape[0]):
		x = index % tiles_w
		y = index // tiles_w

		error = errors[tile]
		if np.isnan(error):
			continue
		fc = [error*255/max_error, 255, 255, 1]
		rect = mpatches.Rectangle((x * image_size, y * image_size), image_size, image_size)
		patches.append(rect)
		colors.append(fc)
			#image[((y + 1) * image_size, x * image_size):(x + 1) * image_size] = canvas[index]
	collection = PatchCollection(patches, alpha=0.35)
	collection.set_facecolors(colors)
	collection.set_edgecolors(colors)
	collection.set_linewidth(0.1)
	return collection

def select_tile(tile, list_index, tile_index, orientation, canvas, indices, success):
	indices[list_index] = tile_index
	success[list_index] = True

	if orientation > 0:
		tile = np.rot90(tile, orientation)
	canvas[list_index, :, :] = tile


def backtrack(failure, current_index, success, indices, tiles_w):
	if failure[0] == current_index:  # there was a previous failure at this location
		num_tiles = len(success)
		failure_count = failure[1]
		total_failure_count = failure[2]
		num_backtrack = failure_count#np.random.randint(failure_count)  # choose how many steps to backtrack according to the number of failures at this location

		if failure_count > failure_threshold or total_failure_count > 2*num_tiles: #if failing too often
			if np.random.rand(1) > 0.8:
				success.fill(0) #restart
				indices.fill(0)
				failure = (0, 0, 0)
				print('failed too often, restart!')
			num_backtrack = tiles_w + 1 #track back a full row

		success[current_index - num_backtrack:] = [False] * (num_tiles - current_index + num_backtrack)
		indices[current_index - num_backtrack:] = [0] * (num_tiles - current_index + num_backtrack)

		failure = (current_index, failure_count + 1, total_failure_count + 1)
		print('no tiles found at tile #{}: failed {} times, backtracking {} tiles'.format(current_index, failure_count, num_backtrack))
	else:
		failure = (current_index, 1, failure[2] + 1)  # set failure index, but do nothing (try again same tile)
	return failure

def generate_map(tiles_w, tiles_h, image_size, canvas, indices, success, errors, fixed_tile_list, images, identifiers, edge_hash, corner_hash):
	num_tiles = tiles_w * tiles_h
	print('{} tiles'.format(num_tiles))
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

	failure = (0, 0, 0)
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
		y = current_index // tiles_w
		#print('[{},{}]'.format(x, y))

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
					best_matches_y.append((tile, orientations[idx], distances_y[idx]))

			num_best = len(best_matches_y)
			if num_best > 0:
				random_tile = best_matches_y[int(random.choice(range(num_best)))] # randomly pick one of selected tiles
				index = int(random_tile[0])
				orientation = int(random_tile[1])
				distance = int(random_tile[2])

				if top_all_water and random.random() > 0.125:
					select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
				elif top_all_land and random.random() > 0.125:
					select_tile(all_land, current_index, num_images+1, 0, canvas, indices, success)
				else:
					# write selected tile to canvas
					select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					errors[x][y] += distance
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
					best_matches_x.append((tile, orientations[idx], distances_x[idx]))

			num_best = len(best_matches_x)
			if num_best > 0:
				random_tile = best_matches_x[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
				index = int(random_tile[0])
				orientation = int( (  random_tile[1] + 1 ) % 4 )
				distance = int(random_tile[2])

				if left_all_water and random.random() > 0.125:
					select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
				elif left_all_land and random.random() > 0.125:
					select_tile(all_land, current_index, num_images+1, 0, canvas, indices, success)
				else:
					# write selected tile to canvas
					select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					errors[x][y] += distance
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
						best_matches.append((tile, orientations[idx], distances[idx]))

				num_best = len(best_matches)
				if num_best > 0:
					random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int(random_tile[1])
					distance = int(random_tile[2])

					if left_all_water and top_all_water and random.random() > 0.075:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and top_all_land and random.random() > 0.075:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
						errors[x][y] += distance
						#errors[x][y] += distance
					continue

		# do stochastic backtracking
		failure = backtrack(failure, current_index, success, indices, tiles_w)
	return canvas, indices, errors

def generate_map_all_edges(tiles_w, tiles_h, image_size, canvas, indices, success, errors, fixed_tile_list, images, identifiers, edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash):
	num_tiles = tiles_w * tiles_h
	num_images = images.shape[0]

	all_water = np.zeros((image_size, image_size))
	all_land = 255 * np.ones((image_size, image_size))

	failure = (0, 0)
	progress_bar = tqdm(total=num_tiles)
	progress = 0

	while not np.all(success):
		current_index = np.argmin(success) # find first occurrence of False in success

		if fixed_tile_list[current_index] == 1:
			success[current_index] = True
			continue

		if current_index > progress:
			progress_bar.update(1)
			progress += 1

		if current_index == 0: # for start tile, randomly initialize tile
			init_start_tile(canvas, images, indices)
			success[current_index] = True
			continue

		x = current_index % tiles_w
		y = current_index // tiles_w

		current_used_idxs = indices[np.nonzero(indices)]

		left_all_water = True
		left_all_land = True
		top_all_water = True
		top_all_land = True

		constrained_top = False
		constrained_bottom = False
		constrained_right = False
		constrained_left = False

		if x != 0:  # if not left border, consider edge with left neighbor
			constrained_left = True
			tile_left_right_edge = get_tile_edge_from_canvas(canvas, image_size, RIGHT, current_index - 1, flip=True)
			left_all_water = np.array_equal(tile_left_right_edge, np.zeros(image_size))
			left_all_land = np.array_equal(tile_left_right_edge, 255*np.ones(image_size))

		if y != 0:  # if not top border, consider edge with top neighbor
			constrained_top = True
			tile_top_bottom_edge = get_tile_edge_from_canvas(canvas, image_size, BOTTOM, current_index - tiles_w, flip=False)
			top_all_water = np.array_equal(tile_top_bottom_edge, np.zeros(image_size))
			top_all_land = np.array_equal(tile_top_bottom_edge, 255*np.ones(image_size))

		if x != tiles_w-1 and fixed_tile_list[y * tiles_w + x + 1] == 1:  # if not right border, consider edge with right neighbor
			tile_right_left_edge = get_tile_edge_from_canvas(canvas, image_size, LEFT, current_index + 1, flip=True)
			right_all_water = np.array_equal(tile_right_left_edge, np.zeros(image_size))
			right_all_land = np.array_equal(tile_right_left_edge, 255*np.ones(image_size))
			constrained_right = True

		if y != tiles_h-1 and fixed_tile_list[(y + 1) * tiles_w + x] == 1:  # if not bottom border, consider edge with bottom neighbor
			tile_bottom_top_edge = get_tile_edge_from_canvas(canvas, image_size, TOP, current_index + tiles_w, flip=True)
			bottom_all_water = np.array_equal(tile_bottom_top_edge, np.zeros(image_size))
			bottom_all_land = np.array_equal(tile_bottom_top_edge, 255*np.ones(image_size))
			constrained_bottom = True

		if constrained_top and not constrained_left and not constrained_right and not constrained_bottom:  # handle first column (disregard left neighbor)
			list_idxs_y, distances_y = get_nearest(edge_hash, tile_top_bottom_edge, num_results=max_results, max_distance=max_distance)
			matches_y, orientations = zip(*identifiers[list_idxs_y])

			best_matches_y = []
			for idx, tile in enumerate(matches_y):  # find all eligible tiles that have similar low distance
				if distances_y[idx] > distances_y[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches_y.append((tile, orientations[idx], distances_y[idx]))

			num_best = len(best_matches_y)
			if num_best > 0:
				random_tile = best_matches_y[int(random.choice(range(num_best)))] # randomly pick one of selected tiles
				index = int(random_tile[0])
				orientation = int(random_tile[1])
				distance = int(random_tile[2])

				if top_all_water and random.random() > 0.125:
					select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
				elif top_all_land and random.random() > 0.125:
					select_tile(all_land, current_index, num_images+1, 0, canvas, indices, success)
				else:
					# write selected tile to canvas
					select_tile(images[index], current_index, index, orientation, canvas, indices, success)	
					errors[current_index] += distance
				continue
		elif constrained_left and not constrained_top and not constrained_right and not constrained_bottom:  # handle first row (disregard top neighbor)
			list_idxs_x, distances_x = get_nearest(edge_hash, tile_left_right_edge, num_results=max_results, max_distance=max_distance)
			matches_x, orientations = zip(*identifiers[list_idxs_x])

			best_matches_x = []
			for idx, tile in enumerate(matches_x):  # find all eligible tiles that have similar low distance
				if distances_x[idx] > distances_x[0] + eps:
					break
				if tile not in current_used_idxs:
					best_matches_x.append((tile, orientations[idx], distances_x[idx]))

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
		elif constrained_left and not constrained_top and constrained_right and not constrained_bottom:  # left and right
			edges = np.concatenate([tile_left_right_edge, tile_right_left_edge])
			list_idxs_x, distances_x = get_nearest(opposite_edges_hash, edges, num_results=max_results,
												   max_distance=max_distance)
			if len(list_idxs) > 0:
				matches_x, orientations = zip(*identifiers[list_idxs_x])

				best_matches_x = []
				for idx, tile in enumerate(matches_x):  # find all eligible tiles that have similar low distance
					if distances_x[idx] > distances_x[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches_x.append((tile, orientations[idx], distances_x[idx]))

				num_best = len(best_matches_x)
				if num_best > 0:
					random_tile = best_matches_x[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int((random_tile[1] + 1) % 4)

					if left_all_water and random.random() > 0.125:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and random.random() > 0.125:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						# write selected tile to canvas
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					continue
		elif not constrained_left and constrained_top and not constrained_right and constrained_bottom:  # top and bottom
			edges = np.concatenate([tile_top_bottom_edge, tile_bottom_top_edge])
			list_idxs_x, distances_x = get_nearest(opposite_edges_hash, edges, num_results=max_results,
												   max_distance=max_distance)
			if len(list_idxs) > 0:
				matches_x, orientations = zip(*identifiers[list_idxs_x])

				best_matches_x = []
				for idx, tile in enumerate(matches_x):  # find all eligible tiles that have similar low distance
					if distances_x[idx] > distances_x[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches_x.append((tile, orientations[idx], distances_x[idx]))

				num_best = len(best_matches_x)
				if num_best > 0:
					random_tile = best_matches_x[
						int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int((random_tile[1] + 1) % 4)

					if left_all_water and random.random() > 0.125:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and random.random() > 0.125:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						# write selected tile to canvas
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					continue

		elif not constrained_right and not constrained_bottom: #regular case
			corner = np.concatenate([tile_left_right_edge, tile_top_bottom_edge])
			list_idxs, distances = get_nearest(corner_hash, corner, num_results=max_results, max_distance=max_distance)
			if len(list_idxs) > 0:
				matches, orientations = zip(*identifiers[list_idxs])

				best_matches = []
				for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
					if distances[idx] > distances[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches.append((tile, orientations[idx], distances[idx]))

				num_best = len(best_matches)
				if num_best > 0:
					random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int(random_tile[1])
					distance = int(random_tile[2])

					if left_all_water and top_all_water and random.random() > 0.075:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and top_all_land and random.random() > 0.075:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
						errors[current_index] += distance
					continue
		elif not constrained_bottom and constrained_right: #three sides constrained
			three_edges = np.concatenate([tile_left_right_edge, tile_top_bottom_edge, tile_right_left_edge])
			list_idxs, distances = get_nearest(three_edges_hash, three_edges, num_results=max_results, max_distance=max_distance)
			if len(list_idxs) > 0:
				matches, orientations = zip(*identifiers[list_idxs])

				best_matches = []
				for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
					if distances[idx] > distances[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches.append((tile, orientations[idx], distances[idx]))

				num_best = len(best_matches)
				if num_best > 0:
					random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int(random_tile[1])
					distance = int(random_tile[2])

					if left_all_water and top_all_water and random.random() > 0.075:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and top_all_land and random.random() > 0.075:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
						errors[current_index] += distance
					continue
		elif not constrained_right and constrained_bottom: #three sides constrained
			three_edges = np.concatenate([tile_bottom_top_edge, tile_left_right_edge, tile_top_bottom_edge])
			list_idxs, distances = get_nearest(three_edges_hash, three_edges, num_results=max_results, max_distance=max_distance)
			if len(list_idxs) > 0:
				matches, orientations = zip(*identifiers[list_idxs])

				best_matches = []
				for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
					if distances[idx] > distances[0] + eps:
						break
					if tile not in current_used_idxs:
						best_matches.append((tile, orientations[idx], distances[idx]))

				num_best = len(best_matches)
				if num_best > 0:
					random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
					index = int(random_tile[0])
					orientation = int(random_tile[1])
					distance = int(random_tile[2])

					if left_all_water and top_all_water and random.random() > 0.075:
						select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
					elif left_all_land and top_all_land and random.random() > 0.075:
						select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
					else:
						select_tile(images[index], current_index, index, orientation, canvas, indices, success)
					continue
			else:  # all sides constrained
				four_edges = np.concatenate([tile_top_bottom_edge, tile_right_left_edge, tile_bottom_top_edge, tile_left_right_edge])
				list_idxs, distances = get_nearest(four_edges_hash, four_edges, num_results=max_results,
														   max_distance=max_distance)
				if len(list_idxs) > 0:
					matches, orientations = zip(*identifiers[list_idxs])

					best_matches = []
					for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
						if distances[idx] > distances[0] + eps:
							break
						if tile not in current_used_idxs:
							best_matches.append((tile, orientations[idx], distances[idx]))

					num_best = len(best_matches)
					if num_best > 0:
						random_tile = best_matches[
							int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
						index = int(random_tile[0])
						orientation = int(random_tile[1])
						distance = int(random_tile[2])

						if left_all_water and top_all_water and random.random() > 0.075:
							select_tile(all_water, current_index, num_images, 0, canvas, indices, success)
						elif left_all_land and top_all_land and random.random() > 0.075:
							select_tile(all_land, current_index, num_images + 1, 0, canvas, indices, success)
						else:
							select_tile(images[index], current_index, index, orientation, canvas, indices, success)					
							errors[current_index] += distance
						continue
		# do stochastic backtracking
		failure = backtrack(failure, current_index, success, indices, tiles_w)
	return canvas, indices, errors

def fill_hole(npimage, pos):
	top    = npimage[pos[1], pos[0]:pos[0]+image_size]  # TOP
	right  = npimage[pos[1]:pos[1]+image_size, pos[0]+image_size]  # RIGHT
	bottom = np.flip(npimage[pos[1]+image_size, pos[0]:pos[0]+image_size], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(npimage[pos[1]:pos[1]+image_size, pos[0]], 0)  # LEFT, flip to preserve clockwise order

	edges = np.concatenate([top, right, bottom, left])
	list_idxs, distances = get_nearest(tile_hash, edges, num_results=max_results, max_distance=max_distance)
	matches, orientations = zip(*identifiers[list_idxs])

	best_matches = []
	for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
		if distances[idx] > distances[0] + eps:
			break

		best_matches.append((tile, orientations[idx], distances[idx]))

	num_best = len(best_matches)
	if num_best > 0:
		random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
		tile = images[int(random_tile[0])]
		orientation = int(random_tile[1])
		distance = int(random_tile[2])
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
	list_idxs, distances = get_nearest_tiles(tile_hash, edges, num_results=max_results, max_distance=max_distance)
	matches, orientations = zip(*identifiers[list_idxs])

	best_matches = []
	for idx, tile in enumerate(matches):  # find all eligible tiles that have similar low distance
		if distances[idx] > distances[0] + eps:
			break

		best_matches.append((tile, orientations[idx], distances[idx]))

	num_best = len(best_matches)
	if num_best > 0:
		random_tile = best_matches[int(random.choice(range(num_best)))]  # randomly pick one of selected tiles
		satellite_tile = sat_images[int(random_tile[0])]
		binary_tile = images[int(random_tile[0])]
		orientation = int(random_tile[1])
		distance = int(random_tile[2])

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
max_distance = 10000  # 1024#
max_results = 10000
eps = 1500
failure_threshold = 10

#######################################################################################################################
## map generating
# target_width = 2048
# target_height = 1024
#
# database_path = 'data/coastlines_binary_cleaned_128_images.hdf5'
# image_size = int(re.search(r'\d+', database_path).group()) #auto-extract integer from string
# hdf5_file = h5py.File(database_path , "r")
#
# images = np.array(hdf5_file["images"])
#
# tiles_w = int(np.floor(target_width / image_size))#30
# tiles_h = int(np.floor(target_height / image_size))#20
#
# single_edge_hash_path = "data/hash_database_" + str(image_size) + "_edges.ann"
# corner_hash_path = "data/hash_database_" + str(image_size) + "_corners.ann"
# opposite_edges_hash_path = "data/hash_database_" + str(image_size) + "_opposite.ann"
# three_edges_hash_path = "data/hash_database_" + str(image_size) + "_three.ann"
# four_edges_hash_path = "data/hash_database_" + str(image_size) + "_four.ann"
#
# #def generate_hashes(images, filename_single_edges, filename_corners, filename_opposite_edges, filename_three_edges, filename_four_edges)
# #return single_edge_index, corner_index, opposite_edges_index, three_edges_index, four_edges_index, identifiers
#
# edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash, identifiers = generate_hashes(images, single_edge_hash_path, corner_hash_path, opposite_edges_hash_path, three_edges_hash_path, four_edges_hash_path)
#
# if len(images.shape) > 3:
# 	canvas = np.zeros((tiles_h * tiles_w, image_size, image_size, images.shape[3]), dtype='uint8')
# else:
# 	canvas = np.zeros((tiles_h * tiles_w, image_size, image_size), dtype='uint8')
#
# fixed_tile_list = np.zeros(tiles_h * tiles_w+1, dtype='uint8')
# indices = np.zeros(tiles_h * tiles_w)
# success = np.zeros(tiles_h * tiles_w, dtype=bool)
#
# #put some random tiles in canvas to try out
# x=2
# y=6
# select_tile(images[286], y*tiles_h+x, 286, 0, canvas, indices, success)
# fixed_tile_list[y*tiles_h+x] = 1
# x=12
# y=4
# select_tile(images[2345], y*tiles_h+x, 2345, 0, canvas, indices, success)
# fixed_tile_list[y*tiles_h+x] = 1
# x=5
# y=5
# select_tile(images[645], y*tiles_h+x, 645, 0, canvas, indices, success)
# fixed_tile_list[y*tiles_h+x] = 1
#
# binary_canvas, indices = generate_map(tiles_w, tiles_h, canvas, indices, success, fixed_tile_list, images)
#
# binary_image = get_image_from_canvas(binary_canvas, tiles_w, tiles_h)
# save_map(binary_image, 'binary')