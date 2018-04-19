#Find approximate nearest neighbor
#@ author: Anna Frühstück

from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
import os.path
#from stitch_tiles_work import get_all_edges_from_array

#######################################################################################################################
# SPOTIFY ANNOY PYTHON LIBRARY
#######################################################################################################################

def generate_hashes(images, filename_corners, filename_edges):
	num_images = images.shape[0]
	#tile_index = AnnoyIndex(4 * images.shape[1], metric='euclidean')
	corner_index = AnnoyIndex(2 * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	edge_index = AnnoyIndex(images.shape[1], metric='euclidean')  # Length of item vector that will be indexed

	#full_filename_tiles = os.path.join(os.getcwd(), filename_tiles.replace('/', '\\'))
	full_filename_corners = os.path.join(os.getcwd(), filename_corners.replace('/', '\\'))
	full_filename_edges = os.path.join(os.getcwd(), filename_edges.replace('/', '\\'))

	identifiers = np.column_stack((np.floor(np.arange(0, num_images, 0.25)), np.tile(range(4), (1, num_images))[0]))

	if os.path.isfile(full_filename_corners) and os.path.isfile(full_filename_edges):# and os.path.isfile(full_filename_tiles):
		corner_index.load(full_filename_corners)
		edge_index.load(full_filename_edges)
		#tile_index.load(full_filename_tiles)

		return corner_index, edge_index, identifiers
	ct = 0
	for idx, image in enumerate(tqdm(images)):
		# for i in xrange(1000):
		(top, right, bottom, left) = get_all_edges_from_array(image)

		edge_index.add_item(ct, top)
		edge_index.add_item(ct + 1, right)
		edge_index.add_item(ct + 2, bottom)
		edge_index.add_item(ct + 3, left)

		corner_left_top = np.concatenate([left, top])
		corner_top_right = np.concatenate([top, right])
		corner_right_bottom = np.concatenate([right, bottom])
		corner_bottom_left = np.concatenate([bottom, left])

		# adding one to each edge to avoid zero vector (which annoy doesn't handle)
		corner_index.add_item(ct, corner_left_top)
		corner_index.add_item(ct + 1, corner_top_right)
		corner_index.add_item(ct + 2, corner_right_bottom)
		corner_index.add_item(ct + 3, corner_bottom_left)

		# tile_edge_top    = np.concatenate([top, right, bottom, left]) #np.concatenate([top, np.concatenate([right, np.concatenate([bottom, left])])])
		# tile_edge_right  = np.concatenate([right, bottom, left, top]) #np.concatenate([right, np.concatenate([bottom, np.concatenate([left, top])])])
		# tile_edge_bottom = np.concatenate([bottom, left, top, right]) #np.concatenate([bottom, np.concatenate([left, np.concatenate([top, right])])])
		# tile_edge_left   = np.concatenate([left, top, right, bottom]) #np.concatenate([left, np.concatenate([top, np.concatenate([right, bottom])])])
		#
		# # adding one to each edge to avoid zero vector (which annoy doesn't handle)
		# tile_index.add_item(ct, tile_edge_top)
		# tile_index.add_item(ct + 1, tile_edge_right)
		# tile_index.add_item(ct + 2, tile_edge_bottom)
		# tile_index.add_item(ct + 3, tile_edge_left)
		ct += 4
	corner_index.build(10)  # 10 trees
	edge_index.build(10)  # 10 trees
	#tile_index.build(10)  # 10 trees

	corner_index.save(filename_corners)
	edge_index.save(filename_edges)
	#tile_index.save(filename_tiles)

	return corner_index, edge_index, identifiers

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

def generate_tile_hash(images, filename):
	num_images = images.shape[0]
	engine = AnnoyIndex(4*images.shape[1], metric='euclidean')  # Length of item vector that will be indexed

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

		tile_edge_top = np.concatenate([top, right, bottom, left])
		tile_edge_right = np.concatenate([right, bottom, left, top])
		tile_edge_bottom = np.concatenate([bottom, left, top, right])
		tile_edge_left = np.concatenate([left, top, right, bottom])

		engine.add_item(ct,   tile_edge_top)
		engine.add_item(ct+1, tile_edge_right)
		engine.add_item(ct+2, tile_edge_bottom)
		engine.add_item(ct+3, tile_edge_left)
		identifiers[ct  ] = [idx, 0]
		identifiers[ct+1] = [idx, 1]
		identifiers[ct+2] = [idx, 2]
		identifiers[ct+3] = [idx, 3]
		ct += 4
	engine.build(10)  # 10 trees
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

def get_nearest_tiles(engine, edge, num_results=10000, max_distance=100000):
	# adding one to edge to avoid zero vector (which annoy doesn't handle)
	(nn_idxs, nn_dists) = engine.get_nns_by_vector(edge, num_results, include_distances=True)
	#filter results by distances
	nearest_tiles_idxs, distances = zip(* [[e, d] for e, d in zip(nn_idxs, nn_dists) if d <= max_distance ])

	return list(nearest_tiles_idxs), list(distances)

# return all edges from image array
def get_all_edges_from_array(image):
	top    = image[0, :]  # TOP
	right  = image[:, image.shape[0]-1]  # RIGHT
	bottom = np.flip(image[image.shape[1]-1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(image[:, 0], 0)  # LEFT, flip to preserve clockwise order
	return top, right, bottom, left