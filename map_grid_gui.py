import pygame as pg
from pygame.locals import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import math
import numpy as np


########################################################################################################################
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
from tile_ANN import generate_hashes, generate_tile_hash, get_nearest
from generate_tiled_map import *
from matplotlib.collections import PatchCollection

########################################################################################################################
# Define some RGB colors
BLACK 		= (000, 000, 000)
LIGHT_GRAY 	= (240, 240, 240)
WHITE 		= (255, 255, 255)
GREEN 		= (000, 255, 000)
RED 		= (255, 000, 000)

# Mouse buttons
LEFT_BUTTON = 1
MIDDLE_BUTTON = 2
RIGHT_BUTTON = 3

# This sets the number of cells in each grid direction
CELLS_H = 10
CELLS_V = 14
WINDOW_HEIGHT = 1500 #width is inferred from height

CELL_SIZE = math.floor(WINDOW_HEIGHT / CELLS_V)
WINDOW_HEIGHT = CELL_SIZE * CELLS_V
WINDOW_WIDTH = CELL_SIZE * CELLS_H

# This sets the margin inside each cell
MARGIN = 2

GRID_COLOR = LIGHT_GRAY
FOREGROUND_COLOR = WHITE

########################################################################################################################

max_distance = 4096  # 1024#
max_results = 10000
eps = 25
failure_threshold = 7

database_path = 'data/coastlines_binary_cleaned_128_images.hdf5'
image_size = int(re.search(r'\d+', database_path).group())  # auto-extract integer from string
hdf5_file = h5py.File(database_path, "r")

images = np.array(hdf5_file["images"])

# tiles_w = int(np.floor(WINDOW_WIDTH / image_size))  # 30
# tiles_h = int(np.floor(WINDOW_HEIGHT / image_size))  # 20

single_edge_hash_path = "data/hash_database_" + str(image_size) + "_edges.ann"
corner_hash_path = "data/hash_database_" + str(image_size) + "_corners.ann"
opposite_edges_hash_path = "data/hash_database_" + str(image_size) + "_opposite.ann"
three_edges_hash_path = "data/hash_database_" + str(image_size) + "_three.ann"
four_edges_hash_path = "data/hash_database_" + str(image_size) + "_four.ann"

# def generate_hashes(images, filename_single_edges, filename_corners, filename_opposite_edges, filename_three_edges, filename_four_edges)
# return single_edge_index, corner_index, opposite_edges_index, three_edges_index, four_edges_index, identifiers

edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash, identifiers = generate_hashes(
	images, single_edge_hash_path, corner_hash_path, opposite_edges_hash_path, three_edges_hash_path,
	four_edges_hash_path)

if len(images.shape) > 3:
	canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size, images.shape[3]), dtype='uint8')
else:
	canvas = np.zeros((CELLS_V * CELLS_H, image_size, image_size), dtype='uint8')

fixed_tile_list = np.zeros(CELLS_V * CELLS_H + 1, dtype='uint8')
indices = np.zeros(CELLS_V * CELLS_H)
success = np.zeros(CELLS_V * CELLS_H, dtype=bool)
########################################################################################################################
# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = np.zeros((CELLS_H, CELLS_V))# []
# for row in range(CELLS_H):
# 	# Add an empty array that will hold each cell
# 	# in this row
# 	grid.append([])
# 	for column in range(CELLS_V):
# 		grid[row].append(0)  # Append a cell

# Initialize pygame
pg.init()
#set application icon
icon = pg.image.load('data/globe_blue32.png')
pg.display.set_icon(icon)
# Set title of screen
pg.display.set_caption("Tile Grid")

root = tk.Tk()
root.withdraw()

# Set the CELL_SIZE and CELL_SIZE of the screen
WINDOW_SIZE = [WINDOW_HEIGHT, WINDOW_WIDTH]#[CELLS_V*CELL_SIZE+(CELLS_V-1)*MARGIN, CELLS_H*CELL_SIZE+(CELLS_H-1)*MARGIN]
screen = pg.display.set_mode(WINDOW_SIZE)

main_image = pg.Surface(WINDOW_SIZE)
main_image.fill(GRID_COLOR)
# Draw the grid
for row in range(CELLS_H):
	for column in range(CELLS_V):
		cell_rect = pg.Rect(CELL_SIZE*column + MARGIN, CELL_SIZE*row + MARGIN, CELL_SIZE-2*MARGIN, CELL_SIZE-2*MARGIN)
		main_image.fill(FOREGROUND_COLOR, rect=cell_rect)

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pg.time.Clock()

########################################################################################################################
# ------------------------------------------------- Main Program Loop ------------------------------------------------ #
########################################################################################################################
while not done:
	for event in pg.event.get():  # User did something
		if event.type == pg.QUIT:  # If user clicked close
			done = True  # Flag that we are done so we exit this loop
		elif event.type == pg.MOUSEBUTTONUP:
			# User clicks the mouse. Get the position
			pos = pg.mouse.get_pos()
			# Change the x/y screen coordinates to grid coordinates
			column = pos[0] // CELL_SIZE
			row = pos[1] // CELL_SIZE
			print("Clicked ", event.button, " at ", pos, ": Grid coordinates: ", row, column)
			if event.button == LEFT_BUTTON:
				# Set that location to one
				grid[row][column] = 1
				fixed_tile_list[row * CELLS_H + column] = 1

				#hide root window (only show file dialog
				root = tk.Tk()
				root.withdraw()
				file_path = filedialog.askopenfilename()

				image = Image.open(file_path)
				image_canvas = np.array(image.resize((image_size, image_size), Image.ANTIALIAS))
				# print(image_canvas.shape)
				# print(len(image_canvas.shape))
				# print(canvas.shape)
				# print(len(canvas.shape))
				if len(image_canvas.shape) > 2 and len(canvas.shape) == 3:
					image_canvas = image_canvas[:, :, 1]
				select_tile(image_canvas, row * CELLS_H + column, -1, 0, canvas, indices, success)

				image = image.resize((CELL_SIZE, CELL_SIZE), Image.ANTIALIAS)

				#myimage = pg.image.load(file_path)#"coastlines_binary_128/14_-0.0309,127.4439_terrain.png")
				#myimage = pg.transform.smoothscale(myimage, (CELL_SIZE, CELL_SIZE))
				pygame_image = pg.image.fromstring(image.tobytes('raw', 'RGB'), (CELL_SIZE, CELL_SIZE), 'RGB')
				main_image.blit(pygame_image, (CELL_SIZE * column, CELL_SIZE * row))
			elif event.button == RIGHT_BUTTON: #remove tile
				# Set that location to zero
				grid[row][column] = 0
				fixed_tile_list[row * CELLS_H + column] = 0
				#select_tile(zeros(), row * CELLS_H + column, -1, 0, canvas, indices, success)

				cell_rect = pg.Rect(CELL_SIZE * column, CELL_SIZE * row, CELL_SIZE, CELL_SIZE)
				main_image.fill(GRID_COLOR, rect=cell_rect)

				cell_rect = pg.Rect(CELL_SIZE * column + MARGIN, CELL_SIZE * row + MARGIN, CELL_SIZE - 2 * MARGIN, CELL_SIZE - 2 * MARGIN)
				main_image.fill(FOREGROUND_COLOR, rect=cell_rect)
			elif event.button == MIDDLE_BUTTON: #generate map
				# put some random tiles in canvas to try out
				binary_canvas, indices = generate_map(CELLS_H, CELLS_V, image_size, canvas, indices, success, fixed_tile_list,
													  images, identifiers, edge_hash, corner_hash, opposite_edges_hash, three_edges_hash, four_edges_hash)
				binary_image = get_image_from_canvas(binary_canvas, image_size, CELLS_H, CELLS_V)
				if len(binary_image.shape) == 2:
					binary_image = np.stack((binary_image, binary_image, binary_image), 2)
				image = Image.fromarray(binary_image)
				image = image.resize(WINDOW_SIZE, Image.ANTIALIAS)
				pygame_image = pg.image.fromstring(image.tobytes('raw', 'RGB'), WINDOW_SIZE, 'RGB')
				main_image.blit(pygame_image, (0, 0))



	# Draw the grid
	# for row in range(CELLS_H):
	# 	for column in range(CELLS_V):
	# 		color = WHITE
	# 		if grid[row][column] == 1:
	# 			color = GREEN

			# pg.draw.rect(screen,
			# 				 color,
			# 				 [(MARGIN + CELL_SIZE) * column,
			# 				  (MARGIN + CELL_SIZE) * row,
			# 				  CELL_SIZE,
			# 				  CELL_SIZE])

	screen.blit(main_image, (0, 0))

	# Limit to 60 frames per second
	clock.tick(60)

	# Go ahead and update the screen with what we've drawn.
	pg.display.flip()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pg.quit()