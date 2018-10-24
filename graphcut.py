import argparse
import time
import maxflow
from PIL import Image
import numpy as np

INT_MAX = 2147483647


def graphcut(A, B, orientation='horizontal', overlap_width=-1):
	if orientation == 'horizontal':
		return graphcut_horizontal(A, B, overlap_width=overlap_width)
	elif orientation == 'vertical':
		result_rot, segments_mask, result_cutline_rot = graphcut_horizontal(np.rot90(A), np.rot90(B), overlap_width=overlap_width)
		return np.rot90(result_rot, 3), np.rot90(segments_mask, 3), np.rot90(result_cutline_rot, 3)


def graphcut_horizontal(A, B, overlap_width=-1):
	A = A.astype(np.int8) #cast arrays to datatype that permits negative numbers (this was a tricky one)
	B = B.astype(np.int8)

	A_rows, A_cols = A.shape[0:2] #height, width
	B_rows, B_cols = B.shape[0:2] #height, width
	channels = A.shape[2]

	assert(A_rows == B_rows);

	if overlap_width == -1:
		overlap_width = int(A_cols / 2)

	xoffset = A_cols - overlap_width

	est_nodes = A_rows * overlap_width #one node per pixel in overlap region
	est_edges = est_nodes * 4 #one edge connecting nodes to neighbors

	g = maxflow.Graph[int](est_nodes, est_edges)

	# Add {est_nodes} (non-terminal) nodes
	node_ids = g.add_grid_nodes((A_rows, overlap_width)) #returns a 2D grid of per-pixel node ids

	#print('setting edge weights...')

	# Set the source/sink weights
	for y in range(A_rows):
		g.add_tedge(y * overlap_width + 0, INT_MAX, 0)  # set capacity from source to node and to sink from node
		g.add_tedge(y * overlap_width + overlap_width - 1, 0, INT_MAX)

	AminusB = A[0:A_rows, xoffset:xoffset + overlap_width, :] - B[0:A_rows, 0:overlap_width, :]
	distanceAB = np.linalg.norm(AminusB, axis=2)  # pixelwise euclidean distance between images in overlap region
	sumRight = np.zeros(node_ids.shape)
	sumRight[ :, :-1] = distanceAB[ :, :-1] + distanceAB[ :, 1:] #calculate sum of adjacent euclidean distances (horizontal)
	sumRight = sumRight.astype(int)

	sumBottom = np.zeros(node_ids.shape)
	sumBottom[:-1, :] = distanceAB[ :-1, :] + distanceAB[1:, :] #calculate sum of adjacent euclidean distances (vertical)
	sumBottom = sumBottom.astype(int)

	structure = np.array([[0, 0, 0],
						  [0, 0, 1],
						  [0, 0, 0]])
	# add edges connecting pixel to right adjacent pixel with weight equals sum of euclidean distances
	g.add_grid_edges(node_ids, sumRight, structure=structure, symmetric=True)

	structure = np.array([[0, 0, 0],
						  [0, 0, 0],
						  [0, 1, 0]])
	# add edges connecting pixel to bottom adjacent pixel with weight equals sum of euclidean distances
	g.add_grid_edges(node_ids, sumBottom, structure=structure, symmetric=True)

	#print('calculate flow...')
	flow = g.maxflow()

	#print("max flow: {}".format(flow))

	result_rows = A_rows
	result_cols = A_cols + B_cols - overlap_width
	#print("result shape: {}x{}x{}".format(result_rows, result_cols, channels))
	#print('creating graphcut result image...')

	segments_mask = np.zeros((result_rows, result_cols), dtype=bool)
	segments_mask[:, xoffset:] = np.ones((B_rows, B_cols), dtype=bool)
	segments_mask[:, xoffset:xoffset+overlap_width] = g.get_grid_segments(node_ids) #pixels where image B (=sink) should be picked

	A_pad = np.zeros((result_rows, result_cols, channels))
	A_pad[:, :A_cols] = A
	B_pad = np.zeros((result_rows, result_cols, channels))
	B_pad[:, xoffset:] = B

	result = np.zeros((A_rows, result_cols, channels))
	result[ segments_mask] = B_pad[ segments_mask]
	result[~segments_mask] = A_pad[~segments_mask]

	# find logical changes (boolean value of adjacent pixels changes): from left to right | from top to down
	changes = (np.roll(segments_mask, 1) ^ segments_mask) | (np.roll(segments_mask, 1, axis=0) ^ segments_mask)
	changes[0, :] = False
	#changes[:, 0] = False
	result_cutline = np.copy(result)
	result_cutline[changes] = [210, 255, 0]

	result = result.astype(np.uint8)
	return result, segments_mask, result_cutline

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'TODO Put description')
	parser.add_argument('filenames', metavar='N', type=str, nargs='+',
						help='input filenames')

	args = parser.parse_args()

	filenames = args.filenames

	print(filenames)

	if len(filenames) < 1: #todo check for length of filenames
		print("usage: ./graphcut.py img1.jpg [optional img2.jpg]")
		quit()

	A_img = Image.open(filenames[0])
	# if max(A_img.width, A_img.height) > 1024:
	# 	A_img = A_img.resize((1024, 1024))
	A = np.array(A_img)

	if len(filenames) == 2:
		B_img = Image.open(filenames[1])
		#if max(B_img.width, B_img.height) > 1024:
		#	B_img = B_img.resize((1024, 1024))
		B = np.array(B_img)
	else:
		B = np.copy(A)

	fn_A = filenames[0].split('/')[-1].split('.')[0]
	fn_B = filenames[1].split('/')[-1].split('.')[0]

	print("image {} | {} x {}".format(fn_A, A.shape[0], A.shape[1]))
	print("image {} | {} x {}".format(fn_B, B.shape[0], B.shape[1]))
	overlap_width = int(A.shape[0]/2)
	start = time.time()

	result, segments_mask, result_cutline = graphcut(A, B, orientation='horizontal', overlap_width=overlap_width)

	end = time.time()
	print('{:.2f} seconds elapsed'.format(end - start))

	out_img = Image.fromarray(np.uint8(result))
	out_img.save("output/graphcut_"+fn_A+"_"+fn_B+"_"+str(overlap_width)+"px.png", "PNG")

	out_img = Image.fromarray(np.uint8(result_cutline))
	out_img.save("output/graphcut_"+fn_A+"_"+fn_B+"_"+str(overlap_width)+"px_cutline.png", "PNG")
