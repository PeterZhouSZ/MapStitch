# Stitch tiles together to create seamless texture
#@ author: Anna Frühstück


# imports
import random
import numpy as np
from tqdm import tqdm
from evaluate_nearest_neighbor_hnsw import get_nearest
from graphcut import *
import matplotlib.pyplot as plt

# constants
TOP = 0
RIGHT = 1#
BOTTOM = 2
LEFT = 3

# parameters
max_distance = 100000  # 1024#
max_results = 512
eps = 2500
failure_threshold = 10

#random.seed(42)

#######################################################################################################################
# helper functions for map generation
#############################################l##########################################################################

def image_array(height, width, RGB=True):
	if RGB:
		return np.zeros((height, width, 3), dtype=np.uint8)
	else:
		return np.zeros((height, width), dtype=np.uint8)

def get_tile_edge_from_canvas(canvas, edge_type, i, flip=False):
	edge = None
	if edge_type == TOP:
		edge = canvas[i, 0, :]  # TOP
	elif edge_type == RIGHT:
		edge = canvas[i, :, -1]  # RIGHT
	elif edge_type == BOTTOM:
		edge = canvas[i, -1, :]  # BOTTOM
	elif edge_type == LEFT:
		edge = canvas[i, :, 0]  # LEFT
	if flip:
		edge = np.flip(edge, 0)
	return edge.flatten('F')

def init_start_tile(canvas, images, indices):
	good_pick = False
	while not good_pick:
		random_index = int(random.choice(range(images.shape[0])))  # initialize top left randomly
		if len(set(images[random_index, :, :, 2].flatten())) < 50: #check if tile is very homogeneous (reject if yes)
			continue
		indices[0, 0] = random_index
		canvas[0, :, :] = images[random_index]
		good_pick = True

def get_image_from_canvas(canvas, image_size, tiles_w, tiles_h):
	image = np.zeros((image_size * tiles_h, image_size * tiles_w, 3), dtype=np.uint8)

	for index in range(canvas.shape[0]):
		x = index % tiles_w
		y = index // tiles_w
		image[y * image_size : (y + 1) * image_size, x * image_size:(x + 1) * image_size] = canvas[index]

	return image

def load_image(filename, image_size):
	return np.array(Image.open(filename).resize((image_size, image_size), Image.ANTIALIAS), dtype=np.uint8)

def get_graphcut_image_from_canvas(canvas, indices, coordinates, image_size, tiles_w, tiles_h):
	level = 11
	directory = "D:\\Data\\ESRI\\"+str(level)+"\\"
	overlap = int(image_size/2)

	channels = 3 if len(canvas.shape) > 3 else 1
	#TODO fix if RGB

	image = np.zeros((image_size * tiles_h, image_size * tiles_w, channels), dtype=np.uint8)
	cutline_image = np.zeros((image_size * tiles_h, image_size * tiles_w, channels), dtype=np.uint8)

	tile_rows = np.zeros((tiles_h, image_size + 2*overlap, image_size * tiles_w, channels), dtype=np.uint8)
	tile_rows_cutline = np.zeros((tiles_h, image_size + 2*overlap, image_size * tiles_w, channels), dtype=np.uint8)

	canvas_with_neighborhood = np.zeros((canvas.shape[0], image_size + 2*overlap, image_size + 2*overlap, channels), dtype=np.uint8)

	for y in range(tiles_h):
		for x in range(tiles_w):
			i = y * tiles_w + x

			coord_x, coord_y = coordinates[indices[i, 0]]
			orientation = indices[i, 1]

			# TODO handle file not found errors (pick mirror tile canvas[i][..., ::-1, :]
			# create cross-shaped neighborhood of tile for lookup
			#   ┌───┬───────────┬───┐ ┐
			#   │TL │     T     │ TR│ │ overlap
			#   ├───┼───────────┼───┤ ┘
			#   │   │           │   │
			#   │   │           │   │
			#   │ L │ canvas[i] │ R │
			#   │   │           │   │
			#   │   │           │   │
			#   ├───┼───────────┼───┤
			#   │BL │     B     │ BR│
			#   └───┴───────────┴───┘

			neighborhood = np.zeros((image_size + 2*overlap, image_size + 2*overlap, channels), dtype=np.uint8)
			try:
				img_T = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y - 1, coord_x), image_size)
			except FileNotFoundError:
				img_T = canvas[i][::-1, ..., :]
			try:
				img_R = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y, coord_x + 1), image_size)
			except FileNotFoundError:
				img_R = canvas[i][ ...,::-1, :]
			try:
				img_B = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y + 1, coord_x), image_size)
			except FileNotFoundError:
				img_B = canvas[i][::-1, ..., :]
			try:
				img_L = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y, coord_x - 1), image_size)
			except FileNotFoundError:
				img_L = canvas[i][ ...,::-1, :]
			try:
				img_TR = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y - 1, coord_x + 1), image_size)
			except FileNotFoundError:
				img_TR = canvas[i][::-1,::-1, :]
			try:
				img_TL = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y - 1, coord_x - 1), image_size)
			except FileNotFoundError:
				img_TL = canvas[i][::-1, ::-1, :]
			try:
				img_BR = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y + 1, coord_x + 1), image_size)
			except FileNotFoundError:
				img_BR = canvas[i][::-1, ::-1, :]
			try:
				img_BL = load_image(directory + '{}_{}_{}.jpg'.format(level, coord_y + 1, coord_x - 1), image_size)
			except FileNotFoundError:
				img_BL = canvas[i][::-1, ::-1, :]

			neighborhood[0:overlap, overlap:-overlap] = img_T[-overlap:,:] #fill in upper extension
			neighborhood[overlap:-overlap, 0:overlap] = img_L[:,-overlap:] #fill in left extension
			neighborhood[-overlap:, overlap:-overlap] = img_B[0:overlap,:] #fill in bottom extension
			neighborhood[overlap:-overlap, -overlap:] = img_R[:,0:overlap] #fill in right extension

			#fill in corners
			neighborhood[0:overlap, -overlap:] = img_TR[-overlap:, 0:overlap]  # fill in top-right corner
			neighborhood[0:overlap, 0:overlap] = img_TL[-overlap:, -overlap:]  # fill in top-left corner
			neighborhood[-overlap:, 0:overlap] = img_BL[0:overlap, -overlap:]  # fill in bottom-left corner
			neighborhood[-overlap:, -overlap:] = img_BR[0:overlap, 0:overlap]  # fill in bottom-right corner

			neighborhood = np.rot90(neighborhood, orientation) #rotate to adjust to rotation of original tile
			neighborhood[overlap:-overlap, overlap:-overlap] = canvas[i] #insert original tile at center
			canvas_with_neighborhood[i, :, :] = neighborhood

			if x == 0: #first image in row: just paste image
				tile_rows[y, :, x * image_size:(x + 1) * image_size] = neighborhood[:, overlap:-overlap]
				tile_rows_cutline[y, :, x * image_size:(x + 1) * image_size] = neighborhood[:, overlap:-overlap]

			else: #fix left seam
				extended_current = neighborhood[:, :-overlap]
				extended_left = canvas_with_neighborhood[i - 1, :, overlap:]

				graphcut_tile, segments_mask, graphcut_tile_cutline = graphcut(extended_left, extended_current, orientation='horizontal', overlap_width=2*overlap)
				tile_rows[y, :, (x - 1) * image_size + overlap:(x + 1) * image_size] = graphcut_tile[:, overlap:] #modify left and current tile based on graphcut
				tile_rows_cutline[y, :, (x - 1) * image_size + overlap:(x + 1) * image_size] = graphcut_tile_cutline[:, overlap:]

				if False: #debug output of neighborhood
					ax = plt.subplot(321)
					ax.imshow(canvas[i - 1], aspect='equal')  # [:,:,:]
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					l_coord_x, l_coord_y = coordinates[indices[i-1, 0]]
					ax.set_title('left tile: ({},{}) orientation: {}'.format(l_coord_x, l_coord_y, indices[i - 1, 1]))
					ax = plt.subplot(322)
					ax.imshow(canvas_with_neighborhood[i - 1], aspect='equal')  # [:,:,:]
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					ax = plt.subplot(323)
					ax.imshow(canvas[i], aspect='equal')  # [:,:,:]
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					ax.set_title('current tile: ({},{}) orientation: {}'.format(coord_x, coord_y, indices[i, 1]))
					ax = plt.subplot(324)
					ax.imshow(neighborhood, aspect='equal')  # [:,:,:]
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					ax = plt.subplot(313)
					ax.imshow(graphcut_tile, aspect='equal')  # [:,:,:]
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					plt.show()

		#image[y * image_size:(y + 1) * image_size, x * image_size:(x + 1) * image_size] = tile_RGB#canvas[index]
		if y == 0:
			image[y * image_size:(y + 1) * image_size, :] = tile_rows[y, overlap:-overlap, :]
			cutline_image[y * image_size:(y + 1) * image_size, :] = tile_rows_cutline[y, overlap:-overlap, :]

		else: #graphcut between rows
			graphcut_row, segments_mask, graphcut_row_cutline = graphcut(tile_rows[y - 1, overlap:, :], tile_rows[y, 0:-overlap, :], orientation='vertical',	overlap_width = 2*overlap)
			#print(graphcut_row.shape)

			A_pad = np.zeros(graphcut_row.shape)
			A_pad[:-overlap, :] = tile_rows_cutline[y - 1, overlap:, :]
			B_pad = np.zeros(graphcut_row.shape)
			B_pad[ overlap:, :] = tile_rows_cutline[y, 0:-overlap, :]

			#get graphcut lines output
			graphcut_row_cutline[ segments_mask] = B_pad[ segments_mask]
			graphcut_row_cutline[~segments_mask] = A_pad[~segments_mask]
			changes = (np.roll(segments_mask, 1) ^ segments_mask) | (np.roll(segments_mask, 1, axis=0) ^ segments_mask)
			changes[0, :] = False
			graphcut_row_cutline[changes] = [210, 255, 0]

			image[y * image_size - overlap:(y + 1) * image_size, :] = graphcut_row[overlap:, :]
			cutline_image[y * image_size - overlap:(y + 1) * image_size, :] = graphcut_row_cutline[overlap:, :]
		#else: #fix seam

	return image, cutline_image

def select_tile(tile, list_index, tile_index, orientation, canvas, indices, success):
	indices[list_index] = [tile_index, orientation]
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
			num_backtrack = min(tiles_w + 1, current_index) #track back a full row

		#print('#tiles{} c_index{} num_back{}'.format(num_tiles, current_index, num_backtrack))
		success[current_index - num_backtrack:] = False
		indices[current_index - num_backtrack:, :] = [0, 0]

		failure = (current_index, failure_count + 1, total_failure_count + 1)
		print('no tiles found at tile #{}: failed {} times, backtracking {} tiles'.format(current_index, failure_count, num_backtrack))
	else:
		failure = (current_index, 1, failure[2] + 1)  # set failure index, but do nothing (try again same tile)
	return failure

def generate_map(tiles_w, tiles_h, canvas, indices, success, errors, images, identifiers, edge_hash, corner_hash, downsampling_factor=0):
	num_tiles = tiles_w * tiles_h
	print('{} tiles'.format(num_tiles))

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

		#indices_only = indices[:, 0]
		current_used_idxs = indices[np.nonzero(indices), 0]

		## extract edges from canvas
		if x != 0:  # if not on left border, consider edge with left neighbor
			constraint_left_edge = get_tile_edge_from_canvas(canvas, RIGHT, current_index - 1, flip=True)

		if y != 0:  # if not on top border, consider edge with top neighbor
			constraint_top_edge = get_tile_edge_from_canvas(canvas, BOTTOM, current_index - tiles_w, flip=False)

		## find adjacent tiles
		if x == 0:  # handle first column (disregard left neighbor)
			identifier_idxs, distances = get_nearest(edge_hash, constraint_top_edge, num_results=max_results, max_distance=max_distance)
		elif y == 0: # handle first row (disregard top neighbor)
			identifier_idxs, distances = get_nearest(edge_hash, constraint_left_edge, num_results=max_results, max_distance=max_distance)
		else: #handle corner cases
			corner = np.concatenate([constraint_left_edge, constraint_top_edge])
			identifier_idxs, distances = get_nearest(corner_hash, corner, num_results=max_results, max_distance=4*max_distance)

			# fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 20))
			# l = np.expand_dims(constraint_left_edge.reshape((64, 3), order='F'), axis=1)
			# t = np.expand_dims(constraint_top_edge.reshape((64, 3), order='F'), axis=1)
			# axes[0].imshow(np.hstack((l, l, l, t, t, t)))
			# axes[1].imshow(canvas[current_index-1])
			# axes[2].imshow(canvas[current_index-tiles_w])
			# plt.show()

		if len(identifier_idxs) > 0:
			matching_tiles, orientations = zip(*identifiers[identifier_idxs])

			acceptable_matches = []
			min_distance = distances[0]
			for idx, tile in enumerate(matching_tiles):  # find all eligible tiles that have similarly low distance
				if distances[idx] > min_distance + eps: # or distances[idx] > max_distance
					break
				#if tile not in current_used_idxs:
				acceptable_matches.append(idx) #(tile, orientations[idx], distances[idx]))

			if len(acceptable_matches) > 0:
				select_idx = random.choice(acceptable_matches) # randomly pick one of selected tiles
				index = matching_tiles[select_idx]
				orientation = orientations[select_idx]
				if y == 0:
					orientation = int((orientation + 1) % 4) #adjust orientation for vertical matches only
				distance = distances[select_idx]

				# write selected tile to canvas
				select_tile(images[index], current_index, index, orientation, canvas, indices, success)
				errors[x][y] += distance
				continue

		# if here, no tile was found - do stochastic backtracking
		failure = backtrack(failure, current_index, success, indices, tiles_w)
	return canvas, indices, errors