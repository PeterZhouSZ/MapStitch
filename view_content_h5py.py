import argparse
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

def view_content_of_h5py(path, num_imgs):
	hdf5_file = h5py.File(path, "r")
	# Total number of samples
	images = hdf5_file["images"]
	print(images.shape)
	aspect_ratio = (2, 3)  # (height, width)

	idx = np.random.randint(0, images.shape[0], size=num_imgs)

	num_rows = math.floor(math.sqrt(num_imgs))
	num_cols = math.ceil(num_imgs / num_rows)

	fig, axes = plt.subplots(num_cols, num_rows, sharex=True, sharey=True, figsize=(14, 14) )
	i = 0
	for ii, ax in zip(idx, axes.flatten()):
		show = images[ii]
		ax.imshow(show, aspect='equal')  # [:,:,:]
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		plt.axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.axis('off')
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='View image contents of h5py file.')
	parser.add_argument('path', metavar='P', type=str, help='file path to the h5py file')
	parser.add_argument('num_imgs', metavar='N', type=int, help='number of images to display')

	args = parser.parse_args()

	view_content_of_h5py(args.path, args.num_imgs)