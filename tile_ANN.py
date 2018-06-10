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

def generate_hashes(images, filename_single_edges, filename_corners, filename_opposite_edges, filename_three_edges, filename_four_edges):
	num_images = images.shape[0]
	#tile_index = AnnoyIndex(4 * images.shape[1], metric='euclidean')
	channels = 1
	if len(images.shape) > 3:
		channels = images.shape[3]

	corner_index = AnnoyIndex(2 * channels * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	full_filename_corners = os.path.join(os.getcwd(), filename_corners.replace('/', '\\'))

	single_edge_index = AnnoyIndex(channels * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	full_filename_edges = os.path.join(os.getcwd(), filename_single_edges.replace('/', '\\'))

	opposite_edges_index = AnnoyIndex(2 * channels * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	full_filename_opposite_edges = os.path.join(os.getcwd(), filename_opposite_edges.replace('/', '\\'))

	three_edges_index = AnnoyIndex(3 * channels * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	full_filename_three_edges = os.path.join(os.getcwd(), filename_three_edges.replace('/', '\\'))

	four_edges_index = AnnoyIndex(4 * channels * images.shape[1], metric='euclidean')  # Length of item vector that will be indexed
	full_filename_four_edges = os.path.join(os.getcwd(), filename_four_edges.replace('/', '\\'))

	identifiers = np.column_stack((np.floor(np.arange(0, num_images, 0.25)), np.tile(range(4), (1, num_images))[0]))

	generate_single_edges = not os.path.isfile(full_filename_edges)
	generate_corners = not os.path.isfile(full_filename_corners)
	generate_opposite_edges = not os.path.isfile(full_filename_opposite_edges)
	generate_three_edges = not os.path.isfile(full_filename_three_edges)
	generate_four_edges = not os.path.isfile(full_filename_four_edges)

	if not generate_single_edges:
		single_edge_index.load(full_filename_edges)
		print('loaded single edge index')
	if not generate_corners:# and os.path.isfile(full_filename_tiles):
		corner_index.load(full_filename_corners)
		print('loaded corner index')
	if not generate_opposite_edges:
		opposite_edges_index.load(full_filename_opposite_edges)
		print('loaded opposite edge index')
	if not generate_three_edges:
		three_edges_index.load(full_filename_three_edges)
		print('loaded three edge index')
	if not generate_four_edges:
		four_edges_index.load(full_filename_four_edges)
		print('loaded four edge index')

	if not generate_corners and not generate_single_edges and not generate_opposite_edges and not generate_three_edges and not generate_four_edges:
		print('found all indices, returning...')
		return single_edge_index, corner_index, opposite_edges_index, three_edges_index, four_edges_index, identifiers #if all are already loaded from file, no generation needed - return from here
	
	ct = 0
	for idx, image in enumerate(tqdm(images)):
		(top, right, bottom, left) = get_all_edges_from_array(image)
		if generate_single_edges:
			single_edge_index.add_item(ct, top)
			single_edge_index.add_item(ct + 1, right)
			single_edge_index.add_item(ct + 2, bottom)
			single_edge_index.add_item(ct + 3, left)

		if generate_corners:
			corner_left_top = np.concatenate([left, top])
			corner_top_right = np.concatenate([top, right])
			corner_right_bottom = np.concatenate([right, bottom])
			corner_bottom_left = np.concatenate([bottom, left])

			corner_index.add_item(ct, corner_left_top)
			corner_index.add_item(ct + 1, corner_top_right)
			corner_index.add_item(ct + 2, corner_right_bottom)
			corner_index.add_item(ct + 3, corner_bottom_left)

		if generate_opposite_edges:
			opposite_left_right = np.concatenate([left, right])
			opposite_top_bottom = np.concatenate([top, bottom])
			opposite_right_left = np.concatenate([right, left])
			opposite_bottom_top = np.concatenate([bottom, top])

			opposite_edges_index.add_item(ct, opposite_left_right)
			opposite_edges_index.add_item(ct + 1, opposite_top_bottom)
			opposite_edges_index.add_item(ct + 2, opposite_right_left)
			opposite_edges_index.add_item(ct + 3, opposite_bottom_top)

		if generate_three_edges:
			three_without_top = np.concatenate([right, bottom, left])
			three_without_right = np.concatenate([bottom, left, top])
			three_without_bottom = np.concatenate([left, top, right])
			three_without_left = np.concatenate([top, right, bottom])

			three_edges_index.add_item(ct, three_without_top)
			three_edges_index.add_item(ct + 1, three_without_right)
			three_edges_index.add_item(ct + 2, three_without_bottom)
			three_edges_index.add_item(ct + 3, three_without_left)

		if generate_four_edges:
			tile_edge_top = np.concatenate([top, right, bottom, left])
			tile_edge_right = np.concatenate([right, bottom, left, top])
			tile_edge_bottom = np.concatenate([bottom, left, top, right])
			tile_edge_left = np.concatenate([left, top, right, bottom])

			four_edges_index.add_item(ct, tile_edge_top)
			four_edges_index.add_item(ct + 1, tile_edge_right)
			four_edges_index.add_item(ct + 2, tile_edge_bottom)
			four_edges_index.add_item(ct + 3, tile_edge_left)

		ct += 4

	if generate_single_edges:
		single_edge_index.build(10)  # 10 trees
		single_edge_index.save(filename_single_edges)
		print('generated and saved single edges index')

	if generate_corners:
		corner_index.build(10)  # 10 trees
		corner_index.save(filename_corners)
		print('generated and saved corner index')

	if generate_opposite_edges:
		opposite_edges_index.build(10)  # 10 trees
		opposite_edges_index.save(filename_opposite_edges)
		print('generated and saved opposite edges index')

	if generate_three_edges:
		three_edges_index.build(10)  # 10 trees
		three_edges_index.save(filename_three_edges)
		print('generated and saved three edges index')

	if generate_four_edges:
		four_edges_index.build(10)  # 10 trees
		four_edges_index.save(filename_four_edges)
		print('generated and saved four edges index')

	return single_edge_index, corner_index, opposite_edges_index, three_edges_index, four_edges_index, identifiers

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

def get_nearest(engine, edge, num_results=10000, max_distance=100000):
	# adding one to edge to avoid zero vector (which annoy doesn't handle)
	(nn_idxs, nn_dists) = engine.get_nns_by_vector(edge, num_results, include_distances=True)
	if len(nn_idxs) == 0:
		return [], []
	#filter results by distances
	nearest_edge_idxs, distances = zip(* [[e, d] for e, d in zip(nn_idxs, nn_dists) if d <= max_distance ])

	return list(nearest_edge_idxs), list(distances)

# return all edges from image array
def get_all_edges_from_array(image):
	top    = image[0, :]  # TOP
	right  = image[:, image.shape[0]-1]  # RIGHT
	bottom = np.flip(image[image.shape[1]-1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(image[:, 0], 0)  # LEFT, flip to preserve clockwise order
	return top.flatten('F'), right.flatten('F'), bottom.flatten('F'), left.flatten('F')

