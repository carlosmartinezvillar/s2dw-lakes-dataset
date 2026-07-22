'''
Get some dataset metrics.
'''
from PIL import Image
import glob
import numpy as np
import argparse
import os

def get_dataset_mean_std(dir_path):
	'''
	Results for split in split.txt:
	TRAINING:
		mean:   [123.30515418 132.73142713 131.5999535  115.50730149]
		stddev: [53.223368   51.52951993 53.70758823 55.61025949]

	VALIDATION:
		mean:   [113.75104301 127.27095985 123.56407809 104.57414313]
		stddev: [45.1786742  45.16800589 48.54369489 55.25355884]

	TESTING:
		mean:   [121.77359522 136.63373091 141.33366653 111.1916976 ]
		stddev: [53.0393665  49.9187143  50.0453408  55.90689547]		
	'''

	chip_ids = glob.glob(f"{dir_path}/*_B0X.tif")

	count = 0

	sum_   = np.zeros(4,dtype=np.float64)
	sum_sq = np.zeros(4,dtype=np.float64)


	for chip in chip_ids:

		bands = Image.open(chip).split()		
		arr = np.stack([np.array(b, dtype=np.float64) for b in bands])

		pixels = arr.reshape(4,-1)
		sum_ += pixels.sum(axis=1)
		sum_sq += (pixels ** 2).sum(axis=1)
		count  += pixels.shape[1]


	#MEAN & STD
	mean = sum_ / count
	std  = np.sqrt(sum_sq / count - mean ** 2)

	print("MEAN/STD for R,G,B,NIR:")
	print(mean)
	print(std)


def get_chip_positive_count(file_path):
	'''
	Uses 'stats.txt' in the unsplit chip directory, produced when chipping in chip.py.
	Results for proportion of positive label pixels is
	Positve/total_pixels = ~0.4692
	'''

	with open(file_path,'r') as fp:
		lines = fp.readlines()

	chip_ids = []
	counts   = []
	for line in lines:
		chip_str, px_count = line.rstrip('\n').split('\t')
		chip_ids.append(chip_ids)
		counts.append(int(px_count))

	percentage_water = np.array(counts).sum() / (len(counts)*256*256)

	print(f"Proportion of water pixels: {round(percentage_water,4)}")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',required=True,help='Dataset (chip) directory.')
	args = parser.parse_args()

	assert os.path.isdir(args.chip_dir), f"Chip dir {args.chip_dir} not found."
	
	if args.chip_dir[-1] == '/':
		args.chip_dir = args.chip_dir[:-1]

	return args


if __name__ == '__main__':

	# ARGV/ARGUMENTS
	args = parse_args()
	data_dir = args.chip_dir

	# Get dataset mean/stddev
	# get_dataset_mean_std(data_dir) # -- i.e.: /../../chips_256_sorted/training, etc.

	# Get proportion of positive/water pixels on chip dataset
	get_chip_positive_count(f"{data_dir}/stats.txt") # -- i.e.: in /../../chips_256/ (before split)

