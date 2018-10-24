#Find approximate nearest neighbor
#@ author: Anna Frühstück

import hnswlib
from tqdm import tqdm
import numpy as np
import os.path
import bisect

#######################################################################################################################
# HNSW PYTHON LIBRARY
#######################################################################################################################

def generate_hashes(images, filename_corners, filename_edges):
	num_images = images.shape[0]
	channels = 1
	if len(images.shape) > 3:
		channels = images.shape[3]

	dims = channels * images.shape[1]

	corner_index = hnswlib.Index(space='l2', dim=2 * dims)  # possible options are l2, cosine or ip
	edge_index = hnswlib.Index(space='l2', dim=dims)  # possible options are l2, cosine or ip

	# HNSW Parameter settings
	ef_construction = 500  # reasonable range: 100-2000
	efSearch = 1600  # reasonable range: 100-2000 #if higher, better recall but longer retrieval time
	M = 64  # reasonable range: 5-100

	corner_index.init_index(max_elements=4 * num_images, ef_construction=ef_construction, M=M)
	edge_index.init_index(max_elements=4 * num_images, ef_construction=ef_construction, M=M)

	full_filename_corners = os.path.join(os.getcwd(), filename_corners.replace('/', '\\'))
	full_filename_edges = os.path.join(os.getcwd(), filename_edges.replace('/', '\\'))

	identifiers = np.column_stack((np.floor(np.arange(0, num_images, 0.25)).astype(int), np.tile(range(4), (1, num_images))[0]))
	edges = np.empty([4 * num_images, dims])
	corners = np.empty([4 * num_images, 2 * dims])

	if os.path.isfile(full_filename_corners) and os.path.isfile(full_filename_edges):
		print('found existing hash files, loading...')
		corner_index.load_index(full_filename_corners)
		edge_index.load_index(full_filename_edges)

		return corner_index, edge_index, identifiers

	ct = 0
	print('generating new hashed indices from data...')

	for idx, image in enumerate(tqdm(images)):
		(top, right, bottom, left) = get_all_edges_from_array(image)

		edges[ct:ct + 4] = [top, right, bottom, left]

		corner_left_top = np.concatenate([left, top])
		corner_top_right = np.concatenate([top, right])
		corner_right_bottom = np.concatenate([right, bottom])
		corner_bottom_left = np.concatenate([bottom, left])

		corners[ct:ct + 4] = [corner_left_top, corner_top_right, corner_right_bottom, corner_bottom_left]
		ct += 4

	print('saving hashed edge indices to file...')
	edge_index.add_items(edges, np.arange(4 * num_images))
	edge_index.set_ef(efSearch)  # higher ef leads to better accuracy, but slower search
	edge_index.save_index(filename_edges)

	print('saving hashed corner indices to file...')
	corner_index.add_items(corners, np.arange(4 * num_images))
	corner_index.set_ef(efSearch)  # higher ef leads to better accuracy, but slower search
	corner_index.save_index(filename_corners)

	return corner_index, edge_index, identifiers


def get_nearest(engine, feature_vector, num_results=1024, max_distance=100000):
	(nearest, distances) = engine.knn_query(feature_vector, num_results)

	if len(nearest) == 0:
		return [], []
	nearest = nearest.flatten()
	distances = distances.flatten()
	idx = bisect.bisect_left(distances, max_distance)
	return nearest[:idx], distances[:idx]


# return all edges from image array
def get_all_edges_from_array(image):
	top    = image[0, :]  # TOP
	right  = image[:, -1]  # RIGHT
	bottom = np.flip(image[-1, :], 0)  # BOTTOM, flip to preserve clockwise order
	left   = np.flip(image[:, 0], 0)  # LEFT, flip to preserve clockwise order
	return top.flatten('F'), right.flatten('F'), bottom.flatten('F'), left.flatten('F')