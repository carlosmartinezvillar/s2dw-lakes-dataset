'''
A script to produce 256x256 image-lable chip pairs from existing
directories of Sentinel-2 images and DynamicWorld V1 matching labels.
'''

import os
import xml.etree.ElementTree as ET
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import multiprocessing as mp
import time
from PIL import Image
import sys
import argparse


class Product():
	def __init__(self,safe_path):
		self.safe_path = safe_path
		self.safe_id = safe_path.split('/')[-1]
		self.tile  = self.safe_id[38:44]
		self.date  = self.safe_id[11:26]
		self.orbit = self.safe_id[33:37]

		self.label_path = None

	def get_datastrip_id(str):
		pass


	def get_dw_id():
		pass


def clean_dynamicworld_borders(src: rio.DatasetReader) -> dict:
	'''
	Take a rasterio DatasetReader for a dynamicworld image and get the indices 
	where non-zeros begin at the top, bottom, left, and right.

	Parameters
	----------
	src: rasterio.DatasetReader
		Dataset reader for a dynamic world array (which has zeroes where S2
		still has data, making it redundant to check for zeroes in the S2 array).

	Returns
	-------
	dict
		dictionary with indices of first non-zero values at top, left, right, 
		bottom

	'''
	top    = 0
	bottom = src.height-1
	left   = 0
	right  = src.width-1

	while(True):
		row = src.read(1,window=rio.windows.Window(0,top,src.width,1))
		if row.sum() == 0:
			top += 1
		else:
			break

	while(True):
		row = src.read(1,window=rio.windows.Window(0,bottom,src.width,1))
		if row.sum() == 0:
			bottom -= 1
		else:
			break

	while(True):
		col = src.read(1,window=rio.windows.Window(left,0,1,src.height))
		if col.sum() == 0:
			left += 1
		else:
			break

	while(True):
		col = src.read(1,window=rio.windows.Window(right,0,1,src.height))
		if col.sum() == 0:
			right -= 1
		else:
			break

	return {'top':top, 'bottom':bottom, 'left':left, 'right':right}


def align_s2_dw(s2_src: rio.DatasetReader,dw_src: rio.DatasetReader) -> Tuple:
	'''
	Do everything: match indices and remove borders.
	'''
	# 1. REMOVE DW NO-DATA BORDERS(~1-2px each side)
	dw_ij = remove_label_borders(dw_src) # <---- THIS CAN BE COMBINED

	# 2. MATCH DW to S2 (DW has ~20px less on each side) 
	# DW ij's (px index) -> DW xy's (coords)
	dw_xy_ul = dw_src.xy(dw_ij['top'],dw_ij['left'],offset='center')
	dw_xy_lr = dw_src.xy(dw_ij['bottom'],dw_ij['right'],offset='center')
	# DW xy's (coords) -> S2 ij's (px index)
	s2_ij = {}
	s2_ij['top'],s2_ij['left']     = s2_src.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_ij['bottom'],s2_ij['right'] = s2_src.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

	# 3. TRIM S2 -- REMOVE S2 TILE OVERLAP & ADJUST DW
	if s2_ij['top'] < 492: #shift top border down
		delta        = 492 - s2_ij['top']
		s2_ij['top'] = 492
		dw_ij['top'] = dw_ij['top'] + delta

	if s2_ij['bottom'] > 10487: #shift bottom border up
		delta           = s2_ij['bottom'] - 10487
		s2_ij['bottom'] = 10487	
		dw_ij['bottom'] = dw_ij['bottom'] - delta

	if s2_ij['left'] < 492: #shift left border right
		delta         = 492 - s2_ij['left']
		s2_ij['left'] = 492	
		dw_ij['left'] = dw_ij['left'] + delta

	if s2_ij['right'] > 10487: #shift right border left
		delta          = s2_ij['right'] - 10487
		s2_ij['right'] = 10487		
		dw_ij['right'] = dw_ij['right'] - delta

	return s2_ij,dw_ij	


def get_strided_windows(borders:dict) -> [Tuple]:
	'''
	Given a dicts of boundaries, returns an array list with tuples (i,j) for block indices i,j and 
	window objects corresponding to the block i,j. This considers only the usable area of the raster,
	defined by the dict of borders. For an array with two rows and two columns of no data (zeros), 
	the blocks are offseted and defined as:

			    left   stride   stride*2
				| 0 0 ..  	      |
				| 0 0... |		  | 
	top      ---+--------+--------+----
		    0 0 |        |        |
		    0 0 | (0, 0) | (0, 1) |
		     .  |        |        |
	stride*1 .  +--------+--------+
		     .  |        |        |
		        | (1, 0) | (1, 1) |
		        |        |        |
	stride*2 ---+--------+--------+---
				|                 |


	Parameters
	----------
	borders: dict
		The dictionary containing the first and last indices of usable data in
		both directions.

	Returns
	-------
	List of [(str,str),Window]. Window object is arg to .read() method in 
	rasterio.DatasetReader.read(). Indices i,j are row,col in original raster.

	'''	
	# number of pixel rows and cols within borders
	n_px_rows = borders['bottom'] + 1 - borders['top']
	n_px_cols = borders['right'] + 1 - borders['left']

	#nr of blocks
	block_rows = (n_px_rows - CHIP_SIZE) // STRIDE + 1
	block_cols = (n_px_cols - CHIP_SIZE) // STRIDE + 1
	N_blocks = block_rows * block_cols

	windows = []
	for k in range(N_blocks):
		i = k // block_cols
		j = k % block_cols
		row_start = i * STRIDE + borders['top']
		col_start = j * STRIDE + borders['left']
		W = Window(col_start,row_start,CHIP_SIZE,CHIP_SIZE)
		windows += [[(str(i),str(j)),W]]

	return windows


def chip_image(product,base_id,index,N):

	# STDOUT
	print(f'[{index+1}/{N}] PROCESSING {base_id}')
	start_time = time.time()

	# LOAD BAND ARRAYS, CLIP, & NORMALIZE
	rgbn = []
	zero = []
	for reader in product.s2_readers:

		# LOAD
		band_array  = reader.read(1)

		# IF ONLY NO DATA, SKIP PRODUCT -- SOME EMPTY ARRAYS!?
		if int(band_array.sum()) == 0:
			print(f"EMPTY BAND ARRAY in {reader.files[0]} -- SKIPPING.")
			return

		# CLIP & NORMALIZE
		zero_mask   = band_array == 0
		high_cutoff = int(np.percentile(band_array[~zero_mask],99))
		# low_cutoff  = int(np.percentile(band_array[~zero_mask],1))
		low_cutoff  = 0
		band_array  = np.clip(band_array,low_cutoff,high_cutoff)
		band_array  = np.round((band_array-low_cutoff)/(high_cutoff-low_cutoff)*255).astype(np.uint8)
		zero.append(zero_mask)
		rgbn.append(band_array)

	# SET WINDOWS
	# s2_borders = {'top': 492, 'bottom': 10487, 'left': 492, 'right': 10487}
	s2_windows = get_strided_windows(product.s2_borders)
	dw_windows = get_strided_windows(product.dw_borders)

	# SPLIT WINDOWS INTO WORKER SECTIONS
	process_share = len(s2_windows) // N_PROC
	leftover      = len(s2_windows) % N_PROC
	start         = [i*process_share for i in range(N_PROC)]
	stop          = [i*process_share+process_share for i in range(N_PROC)]
	stop[-1]      += leftover
	s2_window_chunks = [s2_windows[s0:s1] for s0,s1 in zip(start,stop)]
	dw_window_chunks = [dw_windows[s0:s1] for s0,s1 in zip(start,stop)]

	# THROW WORKERS AT WINDOW SECTIONS
	lock = mp.Lock() #lock to log stuff
	processes = []
	for i in range(N_PROC):
		p = mp.Process(
			target=chip_image_worker,
			args=(rgbn,zero_mask,label_path,s2_window_chunks[i],base_id,lock)
		)
		p.start()
		processes.append(p)

	for p in processes:
		p.join(timeout=60)

	# STDOUT	
	exec_time = time.time() - start_time
	print(f"All workers done ({exec_time:.3f} secs). ")


def chip_image_worker(rgbn,zero_mask,label_path,s2_windows,dw_windows,base_id,lock):

	# Label DatasetReader -- per thread
	lbl_rdr = rio.open(label_path,'r',tiled=True)

	# CHIP INFO
	stats = []

	for k,(rowcol,w) in enumerate(s2_windows):

		# LOAD (WINDOW SECTION) LABEL
		lbl_array = lbl_rdr.read(1,window=dw_windows[k][1])

		# CHECK LABEL NO DATA -- SKIP CHIP 
		if (lbl_array == 0).any():
			continue

		# CHECK WATER/LAND RATIO
		n_water = (lbl_arr==1).sum()
		if n_water < WATER_MIN or n_water > WATER_MAX:
			continue

		# CHECK RGB-NIR NO DATA
		zero_window = zero_mask[w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		if zero_window.sum() > 0:
			continue

		# LOAD RGB
		r_array = rgbn[0][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		g_array = rgbn[1][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		b_array = rgbn[2][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		n_array = rgbn[3][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]

		# SAVE BANDS
		row,col = rowcol
		outfile = f"{base_id}_{row:02}_{col:02}_B0X.tif"
		r = Image.fromarray(r_array)
		g = Image.fromarray(g_array)
		b = Image.fromarray(b_array)
		n = Image.fromarray(n_array)
		Image.merge('RGBA',(r,g,b,n)).save(outfile)

		# SAVE LABEL
		outfile = f'{base_id}_{row:02}_{col:02}_LBL.tif'
		Image.fromarray(lbl_array).save(outfile)

		# STATS/LOG
		stats.append(f'{outfile.split('/')[-1]}\t{n_water}')

	# LOG
	lock.acquire()
	with open(f'{CHIP_DIR}/stats.txt','a') as fp:
		fp.write('\n'.join(stats))
	lock.release()


def parse_args():
	pass


if __name__ == '__main__':
	pass
	# parse_args()

	safe_regex = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE/"
	# band2_regex = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE/GRANULE/*/IMG_DATA/R10m/*_B02_10m.jp2"
	s2_safes    = glob.glob(safe_regex,root_dir=S2_DIR)

	########## FILTER ######################### #only products with matching label


	########## SPLIT AND QUEUE ################
	chunk_size  = 50
	N_chunks    = len(s2_products) // chunk_size
	remainder   = len(s2_products) % chunk_size
	chunk_queue = []
	for i in range(N_chunks):
		chunk_queue.append(s2_products[i*chunk_size:i*chunk_size+chunk_size])
	if remainder != 0:
		chunk_queue.append(s2_products[N_chunks*chunk_size:])


	########## BATCH PROCESS ##################
	for chunk in chunk_queue:

		########## COPY CHUNK #############################
		for safe_path,label_path in chunk:
			# COPY BANDS
			sp.run(["cp","-v","-r",f"{S2_DIR}/{safe_path}",WORK_DIR])

			# COPY LABEL
			sp.run(["cp","-v",f"{LABEL_DIR}/{label_path}.tif",WORK_DIR])

	########## CHIP ####################
	for i,(safe_path,label_path) in enumerate(chunk):
		product = Product(safe_path,label_path)
		chip_image(product,i,len(chunk))
