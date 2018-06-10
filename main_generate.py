
from PIL import Image, ImageDraw
import math
from evaluate_nearest_neighbor_hnsw import generate_hashes, get_nearest
#from tile_ANN import generate_hashes, get_nearest
from generate_tiled_map import *

CELLS_H = 16
CELLS_V = 7
WINDOW_HEIGHT = 1500 #width is inferred from height

CELL_SIZE = math.floor(WINDOW_HEIGHT / CELLS_V)
WINDOW_HEIGHT = CELL_SIZE * CELLS_V
WINDOW_WIDTH = CELL_SIZE * CELLS_H

max_distance = 10000  # 1024#
max_results = 10000
eps = 1500
failure_threshold = 10

database_path = 'data/esri_level_eleven_64x64_1M_images.hdf5'#esri_level_eleven_128x128_images.hdf5'
image_size = int(re.search(r'\d+', database_path).group())  # auto-extract integer from string
hdf5_file = h5py.File(database_path, "r")

images = np.array(hdf5_file["images"])

fixed_tile_list = np.zeros(CELLS_V * CELLS_H, dtype='uint8')
indices = np.zeros(CELLS_V * CELLS_H)
success = np.zeros(CELLS_V * CELLS_H, dtype=bool)
errors = np.zeros((CELLS_H+1, CELLS_V+1))

# tiles_w = int(np.floor(WINDOW_WIDTH / image_size))  # 30
# tiles_h = int(np.floor(WINDOW_HEIGHT / image_size))  # 20
prefix = "hash_esri_level_eleven_1M_" + str(image_size)
single_edge_hash_path = "data/" + prefix + "_edges.bin"
corner_hash_path = "data/" + prefix + "_corners.bin"
opposite_edges_hash_path = "data/" + prefix + "_opposite.ann"
three_edges_hash_path = "data/" + prefix + "_three.ann" 
four_edges_hash_path = "data/" + prefix + "_four.ann"

# def generate_hashes(images, filename_single_edges, filename_corners, filename_opposite_edges, filename_three_edges, filename_four_edges)
# return single_edge_index, corner_index, opposite_edges_index, three_edges_index, four_edges_index, identifiers
print('generate/load hashes')
#edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash, identifiers = generate_hashes(images, single_edge_hash_path, corner_hash_path, opposite_edges_hash_path, three_edges_hash_path, four_edges_hash_path)
corner_hash, edge_hash, identifiers = generate_hashes(images, corner_hash_path, single_edge_hash_path)

if len(images.shape) > 3:
	canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size, images.shape[3]), dtype='uint8')
else:
	canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size), dtype='uint8')

print('generating map now')
# canvas, indices, errors = generate_map(CELLS_H, CELLS_V, image_size, canvas, indices, success, errors, fixed_tile_list,
# 													  images, identifiers, edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash)

canvas, indices, errors = generate_map(CELLS_H, CELLS_V, image_size, canvas, indices, success, errors, fixed_tile_list, images, identifiers, edge_hash, corner_hash)

image_array = get_image_from_canvas(canvas, image_size, CELLS_H, CELLS_V)
image = Image.fromarray(image_array)

timestr = time.strftime("%m_%d_%H%M")
image.save("output/arcgis_"+str(image_size)+"_"+str(CELLS_H)+"x"+str(CELLS_V)+"_"+timestr+".png", "PNG")

errors = errors - np.amax(errors)/2
errors = 1/(1+np.exp(-errors))
print(errors)
draw = ImageDraw.Draw(image)

for y in range(CELLS_V): 
    for x in range(CELLS_H):
        hue = int((1 - errors[x][y]) * 120)
        color = 'hsl(%d, %d%%, %d%%)' % (hue, 100, 50) #higher errors go red, lower errors go blue
        
        draw.line((x*image_size, y*image_size, (x+1)*image_size, y*image_size), fill=color)
        draw.line((x*image_size, y*image_size, x*image_size, (y+1)*image_size), fill=color)
image.save("output/arcgis_"+str(image_size)+"_"+str(CELLS_H)+"x"+str(CELLS_V)+"_"+timestr+"_errors.png", "PNG")

#image.show()