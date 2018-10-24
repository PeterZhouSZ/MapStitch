import glob
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import re
from lonlat2color import lonlat2rgba

font_size = 10
mp.rcParams['font.size'] = font_size
mp.rcParams['axes.labelsize'] = font_size
mp.rcParams['axes.linewidth'] = font_size / 12.
mp.rcParams['axes.titlesize'] = font_size
mp.rcParams['legend.fontsize'] = font_size
mp.rcParams['xtick.labelsize'] = font_size
mp.rcParams['ytick.labelsize'] = font_size

f = plt.figure()
my_map = Basemap(projection='cyl', resolution='i')

my_map.drawcoastlines(linewidth=0.3, color='lightgray')
#my_map.drawcountries()
my_map.fillcontinents(color='whitesmoke')
#my_map.drawmapboundary()

#my_map.drawmeridians(np.arange(0, 360, 30))
#my_map.drawparallels(np.arange(-90, 90, 30))

data_dir = 'coastlines_terrain/'
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
colors = [ lonlat2rgba(lon, lat) for lon, lat in zip(lons, lats) ]
#my_map.plot(x, y, c=colors, markersize=1)
sc = plt.scatter(x, y, c=colors, s=0.05, alpha=0.8, zorder=20, lw=0)

#pp = PdfPages('output/map.pdf')
#pp.savefig(sc)
plt.show()

f.savefig("results/map.pdf", bbox_inches='tight')