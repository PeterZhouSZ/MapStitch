
 
import math
from evaluate_nearest_neighbor_hnsw import generate_hashes, get_nearest
from PIL import Image, ImageDraw
#from tile_ANN import generate_hashes, get_nearest
from generate_tiled_map import *

TARGET_WIDTH = 2048
TARGET_HEIGHT = 1024

#CELLS_H = 40
#CELLS_V = 25
#WINDOW_HEIGHT = 1500 #width is inferred from height
#CELL_SIZE = math.floor(WINDOW_HEIGHT / CELLS_V)
#WINDOW_HEIGHT = CELL_SIZE * CELLS_V
#WINDOW_WIDTH = CELL_SIZE * CELLS_H

max_distance = 10000  # 1024#
max_results = 10000
eps = 100
failure_threshold = 10

tile_size = 32
image_type_prefix = '1M'#'desert'
database_path = 'D:/Data/satellite/esri_level_eleven_{}x{}_{}_images.hdf5'.format(tile_size, tile_size, image_type_prefix) #esri_level_eleven_128x128_images.hdf5'
database_large_path = 'D:/Data/satellite/esri_level_eleven_{}x{}_{}_images.hdf5'.format(2*tile_size, 2*tile_size, image_type_prefix) #esri_level_eleven_128x128_images.hdf5'
image_size = int(re.search(r'\d+', database_path).group())  # auto-extract integer from string
print('loading data from hdf5 file...')

CELLS_V = math.ceil(TARGET_HEIGHT / image_size)
CELLS_H = math.ceil(TARGET_WIDTH / image_size)

hdf5_file = h5py.File(database_path, "r")
hdf5_large_file = h5py.File(database_large_path, "r")

images = hdf5_file["images"] #np.array()
images_large = hdf5_large_file["images"] #np.array()
image_size_large = images_large[0].shape[0]

#print('generating at resolution {}x{}, upscaling to resolution {}x{}'.format(image_size, image_size, image_size_large, image_size_large))
print('generating {}x{} tile image at resolution {}x{}'.format(CELLS_H, CELLS_V, image_size, image_size))
fixed_tile_list = np.zeros(CELLS_V * CELLS_H, dtype='uint8')
indices = np.zeros(CELLS_V * CELLS_H)
success = np.zeros(CELLS_V * CELLS_H, dtype=bool)
errors = np.zeros((CELLS_H+1, CELLS_V+1))

downsample_edges = 0
prefix = "hash_esri_level_eleven_{}_".format(image_type_prefix) + str(image_size)

single_edge_hash_path = "data/" + prefix + "_edges.bin"
corner_hash_path = "data/" + prefix + "_corners.bin"
#opposite_edges_hash_path = "data/" + prefix + "_opposite.ann"
#three_edges_hash_path = "data/" + prefix + "_three.ann" 
#four_edges_hash_path = "data/" + prefix + "_four.ann"

print('generate/load hashes')
#edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash, identifiers = generate_hashes(images, single_edge_hash_path, corner_hash_path, opposite_edges_hash_path, three_edges_hash_path, four_edges_hash_path)
corner_hash, edge_hash, identifiers = generate_hashes(images, corner_hash_path, single_edge_hash_path, downsampling_factor=downsample_edges)

for s in range(5): #number of result images
    if len(images.shape) > 3:
        canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size, images.shape[3]), dtype='uint8')
        canvas_large = np.zeros((CELLS_V * CELLS_H, image_size_large, image_size_large, images.shape[3]), dtype='uint8')
    else:
        canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size), dtype='uint8')

    print('generating map now')

    canvas, canvas_large, indices, errors = generate_large_map(CELLS_H, CELLS_V, image_size, canvas, canvas_large, indices, success, errors, fixed_tile_list, images, images_large, identifiers, edge_hash, corner_hash)
    #canvas, indices, errors = generate_map(CELLS_H, CELLS_V, image_size, canvas, indices, success, errors, fixed_tile_list, images, identifiers, edge_hash, corner_hash)

    image_array = get_image_from_canvas(canvas, image_size, CELLS_H, CELLS_V)
    image = Image.fromarray(image_array)

    timestr = time.strftime("%m_%d_%H%M")
    image.save("output/arcgis_"+str(image_size)+"_"+str(CELLS_H)+"x"+str(CELLS_V)+"_"+timestr+".png", "PNG")

    # image_array_large = get_image_from_canvas(canvas_large, image_size_large, CELLS_H, CELLS_V)
    # image_large = Image.fromarray(image_array_large)
    # image_large.save("output/arcgis_"+str(image_size)+"_"+str(CELLS_H)+"x"+str(CELLS_V)+"_"+timestr+"_large.png", "PNG")


    # # errors = errors - np.amax(errors)/2
    # # errors = 1/(1+np.exp(-errors))
    # # print(errors)
    # draw = ImageDraw.Draw(image)
    # maximum_error = math.sqrt(image_size * 3 * 65025) #maximal possible euclidean distance of two vectors of size image_size*3
    # print(maximum_error)
    # for y in range(CELLS_V): 
    #     for x in range(CELLS_H):
    #         current_index = y * CELLS_H + x
            
    #         if x < CELLS_H - 1:
    #             current_tile_right_edge =  get_tile_edge_from_canvas(canvas, image_size, RIGHT, current_index, flip=True)
    #             right_tile_left_edge = get_tile_edge_from_canvas(canvas, image_size, LEFT, current_index + 1, flip=False)
            
    #             error_right = np.linalg.norm([current_tile_right_edge, right_tile_left_edge])
    #             print(error_right)
    #             error_right = error_right/maximum_error #euclidean distance normalized by maximum error
    #             hue_right = int((1 - error_right) * 120)
    #             color_right = 'hsl(%d, %d%%, %d%%)' % (hue_right, 100, 50) #higher errors go red, lower errors go blue     
    #             draw.line(((x+1)*image_size, (y)*image_size, (x+1)*image_size, (y+1)*image_size), fill=color_right)

    #         if y < CELLS_V - 1:
    #             current_tile_bottom_edge =  get_tile_edge_from_canvas(canvas, image_size, BOTTOM, current_index, flip=True)
    #             bottom_tile_top_edge =  get_tile_edge_from_canvas(canvas, image_size, RIGHT, current_index + CELLS_H, flip=False)

    #             error_bottom = np.linalg.norm([current_tile_bottom_edge, bottom_tile_top_edge]) #euclidean distance
    #             print(error_bottom)
    #             error_bottom = error_bottom/maximum_error #euclidean distance normalized by maximum error
    #             hue_bottom = int((1 - error_bottom) * 120)
    #             color_bottom = 'hsl(%d, %d%%, %d%%)' % (hue_bottom, 100, 50) #higher errors go red, lower errors go blue
    #             draw.line(((x)*image_size, (y+1)*image_size, (x+1)*image_size, (y+1)*image_size), fill=color_bottom)
    # image.save("output/arcgis_"+str(image_size)+"_"+str(CELLS_H)+"x"+str(CELLS_V)+"_"+timestr+"_errors.png", "PNG")

#image.show()