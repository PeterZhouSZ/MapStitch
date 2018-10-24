
import time
import re
import h5py
import math
from PIL import Image, ImageDraw
from evaluate_nearest_neighbor_hnsw import generate_hashes
from generate_util import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate tiling of ESRI tiles.')
	parser.add_argument('path', metavar='P', type=str, help='file path to the h5py file')
	parser.add_argument('width', metavar='N', type=int, help='output width')
	parser.add_argument('height', metavar='N', type=int, help='output height')
	parser.add_argument('num_imgs', metavar='N', type=int, help='number of images to generate')

	args = parser.parse_args()

	num_result_images = args.num_imgs#5
	database_path = args.path
	TARGET_WIDTH = args.width #1024
	TARGET_HEIGHT = args.height #512

	#tile_size = 64
	#database_path = 'D:/Data/satellite/esri_level_eleven_{}x{}_1M_images.hdf5'.format(tile_size, tile_size)
	#database_path = 'D:\\Data\\satellite\\esri_level_eleven_64x64_desert_images.hdf5'
	#database_path = 'C:\\Users\\fruehsa\\Projects\\MapStitch\\data\\esri_level_eleven_new_64x64_images.hdf5'
	#database_path = 'C:\\Users\\fruehsa\\Projects\\MapStitch\\data\\esri_eleven_{}x{}.hdf5'.format(tile_size, tile_size)

	image_size = int(re.search(r'\d+', database_path).group())  # auto-extract integer from string

	print('loading data from hdf5 file...')

	CELLS_V = math.ceil(TARGET_HEIGHT / image_size)
	CELLS_H = math.ceil(TARGET_WIDTH / image_size)

	hdf5_file = h5py.File(database_path, "r")

	images = hdf5_file["images"]
	coordinates = hdf5_file["coordinates"]

	print('generating {}x{} tiled image at resolution {}x{}'.format(CELLS_H, CELLS_V, image_size, image_size))

	prefix = "data\\" + database_path.split('\\')[-1].split('.')[0] ##"hash_esri_{}".format(image_size)

	single_edge_hash_path = prefix + "_edges.bin"
	corner_hash_path = prefix + "_corners.bin"

	print('generate/load hashes')
	corner_hash, edge_hash, identifiers = generate_hashes(images, corner_hash_path, single_edge_hash_path)

	for s in range(num_result_images):  # do {number of result images} times
		canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size, images.shape[3]), dtype='uint8')
		indices = np.zeros((CELLS_V * CELLS_H, 2), dtype=np.uint32)
		success = np.zeros((CELLS_V * CELLS_H), dtype=bool)
		errors = np.zeros((CELLS_H + 1, CELLS_V + 1))

		print('generating map now')

		canvas, indices, errors = generate_map(CELLS_H, CELLS_V, canvas, indices, success, errors, images, identifiers, edge_hash, corner_hash, downsampling_factor=4)
		#print(indices)
		image_array = get_image_from_canvas(canvas, image_size, CELLS_H, CELLS_V)
		image = Image.fromarray(image_array)

		timestr = time.strftime("%m_%d_%H%M%S")
		image.save("output/{}_{}_{}x{}.png".format(timestr, image_size, CELLS_H, CELLS_V), "PNG")

		image_array, image_cutline_array = get_graphcut_image_from_canvas(canvas, indices, coordinates, image_size, CELLS_H, CELLS_V)
		image = Image.fromarray(image_array)
		image_cutline = Image.fromarray(image_cutline_array)

		image.save("output/{}_{}_{}x{}_graphcut.png".format(timestr, image_size, CELLS_H, CELLS_V), "PNG")
		image_cutline.save("output/{}_{}_{}x{}_graphcut_cutline.png".format(timestr, image_size, CELLS_H, CELLS_V), "PNG")
