#!/usr/bin/python

import urllib.request, urllib.error, urllib.parse
from io import BytesIO
from PIL import Image
from math import log, exp, tan, atan, pi, ceil
import os, sys
from gmap_utils import *
import os
from datetime import datetime				
from time import sleep

import time
import random
import cv2
import numpy as np

WATER_MIN_TERRAIN = np.array([158, 199, 250], dtype=np.uint8) #minimum value of blue pixel in RGB order
WATER_MAX_TERRAIN = np.array([211, 235, 255], dtype=np.uint8) #maximum value of blue pixel in RGB order

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0


def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon

	
def download_tiles(zoom, lat_start, lat_stop, lon_start, lon_stop, keyfile, size=640, labels=False, maptype='terrain'):
	# Set some important parameters
	scale = 1
	maxsize = 640
	
	# convert coordinates to pixels
	ulx, uly = latlontopixels(lat_start, lon_start, zoom)
	lrx, lry = latlontopixels(lat_stop, lon_stop, zoom)
  
	# calculate total pixel dimensions of final image
	dx, dy = lrx - ulx, uly - lry

	# calculate rows and columns
	cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))
	
	print("%d rows, %d columns" % (rows, cols) )
	# calculate pixel dimensions of each small image
	bottom = 120
	width = int(ceil(dx/cols))
	height = int(ceil(dy/rows))
	alturaplus = height + bottom
	
	user_agent = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1'
	headers = { 'User-Agent' : user_agent }
	
	directory = os.path.join(os.getcwd(), datetime.now().strftime('%Y-%m-%d_%H-%M'))
	try:
		os.makedirs(directory)
	except OSError as e:
		#if e.errno != errno.EEXIST:
		raise  # This was not a "directory exist" error..

	api_key_1 = getAPIKeyFromFile(keyfile, 1)
	api_key_2 = getAPIKeyFromFile(keyfile, 2)

	#final = Image.new("RGB", (int(dx), int(dy)))
	for x in range(cols):
		for y in range(rows):
			dxn = width * (0.5 + x)
			dyn = height * (0.5 + y)
			latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom)
			position = ','.join((str(latn), str(lonn)))

			url = "http://maps.googleapis.com/maps/api/staticmap?center=%4.6f,%4.6f&size=%dx%d&zoom=%d&maptype=%s" % (latn, lonn, size, size, zoom, maptype)
			url_terrain = "http://maps.googleapis.com/maps/api/staticmap?center=%4.6f,%4.6f&size=%dx%d&zoom=%d&maptype=terrain" % (latn, lonn, size, size, zoom)
			#if not labels:
			#	url = url + "&style=feature:all|element:labels|visibility:off"
			
			#remove all styles
			#if not labels:
			#	url = url + "&style=element:labels%7Cvisibility:off&style=feature:administrative%7Cvisibility:off&style=feature:administrative.neighborhood%7Cvisibility:off&style=feature:poi%7Cvisibility:off&style=feature:road%7Cvisibility:off&style=feature:transit.line%7Cvisibility:off&style=feature:transit.station%7Cvisibility:off"
			url_terrain = url_terrain + "&style=element:labels%7Cvisibility:off&style=feature:administrative%7Cvisibility:off&style=feature:administrative.neighborhood%7Cvisibility:off&style=feature:poi%7Cvisibility:off&style=feature:road%7Cvisibility:off&style=feature:transit.line%7Cvisibility:off&style=feature:transit.station%7Cvisibility:off"
			
			#attach API key
			url = url + "&key=" + api_key_1
			url_terrain = url_terrain + "&key=" + api_key_2

			filename = "%d_%3.4f,%3.4f_%s.png" % (zoom, latn, lonn, maptype)
			filename_terrain = "%d_%3.4f,%3.4f_terrain.png" % (zoom, latn, lonn)
			#print(filename)
			
			filename_complete = os.path.join(directory, filename)
			filename_terrain_complete = os.path.join(directory, filename_terrain)

			if not os.path.exists(filename):
				try:
					# if maptype=='terrain':
					# 	buffer_terrain = buffer
					# else:
					buffer_terrain = BytesIO(urllib.request.urlopen(url_terrain).read())
				except Exception as e:
					print("--", filename, "->", e)
					sys.exit(1)
				
				print("-- saving", filename)

				image_terrain = Image.open(buffer_terrain)
				
				cimage = image_terrain.convert("RGB")
				npimage = np.array(cimage)
				dst = cv2.inRange(npimage, WATER_MIN_TERRAIN, WATER_MAX_TERRAIN)
				water = cv2.countNonZero(dst)
				water_percent = 100 * water / (size*size)
				print('%2.2f%% water' % water_percent)

				if water_percent > 3 and water_percent < 98: #skip no water or all water to find coast lines
					try:
						buffer = BytesIO(urllib.request.urlopen(url).read())
					except Exception as e:
						print("--", filename, "->", e)
						sys.exit(1)

					image = Image.open(buffer)

					cropped_satellite = image.crop((0, 0, width, height - 25))
					cropped_satellite.save(filename_complete)

					cropped_terrain = image_terrain.crop((0, 0, width, height - 25))
					cropped_terrain.save(filename_terrain_complete)
				#sleep(0.01)

def getAPIKeyFromFile( filename, idx ):
	with open(filename, 'r') as file:
		data = file.readlines()
		data = [l.replace('\n', '') for l in data]
	return data[idx]

if __name__ == "__main__":
	zoom = 15
	size = 640
	#lats_start = [  -34.378211, -45.480707,  49.939950, 37.454265, 16.724174,  38.860314, -40.236435, -18.495172,  18.347437, 60.315460,   ]
	#lons_start = [  172.387544, 166.388777, -56.014554, -8.346962, 97.022732, -123.667852, -73.937083, -70.663753, 119.625145, -142.352609, ]
	#lats_end   = [  -37.263862, -46.238150,  48.359212, 35.725452,  8.584933,  34.574843, -53.857537, -27.542309,   5.433588,  52.096103,   ]
	#lons_end   = [  176.100923, 167.146834, -52.872464, 11.428428, 98.692654, -120.503823, -71.259533, -70.400081, 126.920066, -128.384266,]

	lats_start = [    2.797655, 38.153457]
	lons_start = [  120.834965, 23.984985]
							#CALIFORNIA  #CHILE COAST  #ALASKA   #GREEK ISLANDS:#17.627008, -63.065185#HAWAII: 20.274436, -156.227417#OAHU: 21.710227, -158.290100#MAUI: 21.047817, -156.723175#KAUAI: 22.251447, -159.793854#-5.324124, 134.086304#GOOD:3.396047, 105.495758#11.449742, 113.997803#-7.719837, 127.521973#10.621467, 119.304199#12.413442, 123.459778##ISLANDS_SEASIA-2.960336, 107.493599#KAUST#23.108470, 38.733215
	lats_end   = [  -4.754361, 36.175020]
	lons_end   = [   135.073246, 27.907104]
		#lat_stop, lon_stop = ##CALIFORNIA##CHILE COAST##ALASKA: ##GREEK ISLANDS:#11.849879, -60.516357#HAWAII:18.825067, -154.645386#OAHU: 21.237382, -157.633667#MAUI: 20.525271, -155.948639#KAUAI: 21.864526, -159.274750#-6.995267, 134.976196#GOOD:`2.707637, 106.621856#9.478186, 117.870483#-8.274689, 130.191651#1.914385, 121.237793#11.672074, 124.866028##ISLANDS_SEASIA-3.021165, 107.548420#KAUST#22.246840, 39.145202
	maptype = 'satellite' #"roadmap", "terrain", "hybrid", "satellite"
	labels = False
	keyfile = 'apikeys.txt'

	for i in range(len(lats_start)):
		lat_start = lats_start[i]
		lon_start = lons_start[i]
		lat_end = lats_end[i]
		lon_end = lons_end[i]
		download_tiles(zoom, lat_start, lat_end, lon_start, lon_end, keyfile, size, labels, maptype)
