import glob
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import re

# make sure the value of resolution is a lowercase L,
#  for 'low', not a numeral 1
my_map = Basemap(projection='robin', lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='gray')
my_map.drawmapboundary()

my_map.drawmeridians(np.arange(0, 360, 30))
my_map.drawparallels(np.arange(-90, 90, 30))

data_dir = 'coastlines_256/'
files_in_directory = list(glob.glob(data_dir + '*.png'))
coordinates = np.empty((len(files_in_directory), 2))
for idx, filename in enumerate(tqdm(files_in_directory)):
	#14_-0.0309,130.7346_terrain
	filename = filename.split('\\')[1]
	m = re.match(r'14_(.*),(.*)_terrain.png', filename)
	lat = float(m.group(1))
	lon = float(m.group(2))
	coordinates[idx] = (lat, lon)

lats, lons = zip(* coordinates)
x, y = my_map(lons, lats)
my_map.plot(x, y, 'bo', markersize=1)

plt.show()