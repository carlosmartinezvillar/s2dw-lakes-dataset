'''
Run a training/validation/testing split.

1.  Testing training/validation split:
	1.1 3/6 UTM zones at random, without replacement.
	1.2 1 tile at random for each UTM zone. This is the test set.
2.  Validation training split:
	2.1 Of the remaining, 3 tiles at random. This is the validation set. 
	2.2 Remainder set as training.
'''
import argparse
import os
import numpy as np
import random
import glob
import shutil


DEFAULT_SEED = 476 #reproduce what's split.txt
NR_TILES     = 3   #nr for validation and testing

def build_tree(chip_names):
	'''
	{zone}
	|-> {tile}
		|-> {raster}
			|--> [chips]
	'''
	tree = {}

	# Iterate thru chips
	for chip in chip_names:
		#Get strings
		raster = chip[0:-11]
		tile   = chip[-16:-11]
		zone   = chip[-16:-13]

		# Append to tree
		if zone in tree:
			if tile in tree[zone]:
				if raster in tree[zone][tile]:
					tree[zone][tile][raster].append(chip)
				else:
					tree[zone][tile][raster] = []
					tree[zone][tile][raster].append(chip)
			else:
				tree[zone][tile] = {raster:[chip]}
		else:
			tree[zone] = {tile:{raster:[chip]}}

	return tree


if __name__ == '__main__':

	#CHECK DIR
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',default=None,required=True,
		help='Dataset directory.')
	args = parser.parse_args()
	assert os.path.isdir(args.chip_dir), "split.py: Incorrect data dir argument."

	#FIX SEEDS
	np.random.seed(DEFAULT_SEED)
	random.seed(DEFAULT_SEED)

	#GET CHIP NAMES
	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=args.chip_dir))
	chips       = [_[0:-8] for _ in band_files] #19456 X 53

	#TREE
	tree = build_tree(chips)

	#GET TEST TILES
	test_zones = np.random.choice(sorted(tree),NR_TILES,replace=False)
	test_tiles = []
	for z in test_zones:
		test_tiles.append(np.random.choice(sorted(tree[z]),1,replace=False)[0])

	#GET VALIDATION+TRAIN TILES
	trainvalidate_tiles = []
	for z in tree:
		for t in tree[z]:
			if t not in test_tiles:
				trainvalidate_tiles.append(t)
	validate_tiles = np.random.choice(sorted(trainvalidate_tiles),NR_TILES,replace=False)
	training_tiles = set(trainvalidate_tiles).difference(validate_tiles)

	#LOG SPLIT
	print(f"TEST TILES:       {test_tiles}")
	print(f"VALIDATION TILES: {validate_tiles}")
	print(f"TRAIN TILES:      {training_tiles}")
	with open('./split.txt','w+') as fp:
		fp.write(f"TEST TILES:       {test_tiles}\n")
		fp.write(f"VALIDATION TILES: {validate_tiles}\n")
		fp.write(f"TRAIN TILES:      {training_tiles}\n")

	#SAVE IN SEPARATAE FOLDERS
	chip_dir = args.chip_dir
	if chip_dir[-1] == '/':
		chip_dir = chip_dir[:-1]
	# new_dir = '/'.join([*chip_dir.split('/')[0:-1],'chips_sorted'])
	new_dir = chip_dir + '_sorted'
	os.mkdir(new_dir)
	os.mkdir(f'{new_dir}/training')
	os.mkdir(f'{new_dir}/validation')
	os.mkdir(f'{new_dir}/testing')

	print("COPYING VALIDATION FILES")
	for tile in validate_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.chip_dir)
		for file in tile_files:
			shutil.copy(f"{args.chip_dir}/{file}",f"{new_dir}/validation/{file}",follow_symlinks=False)

	print("COPYING TESTING FILES")
	for tile in test_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.chip_dir)
		for file in tile_files:
			shutil.copy(f"{args.chip_dir}/{file}",f"{new_dir}/testing/{file}",follow_symlinks=False)

	print("COPYING TRAINING FILES")
	for tile in training_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.chip_dir)
		for file in tile_files:
			shutil.copy(f"{args.chip_dir}/{file}",f"{new_dir}/training/{file}",follow_symlinks=False)

	