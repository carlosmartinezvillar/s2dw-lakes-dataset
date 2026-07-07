'''
A script to produce 256x256 image-lable chip pairs from existing
directories of Sentinel-2 images and DynamicWorld V1 matching labels.
'''
import os
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import glob
import math
import multiprocessing as mp
import time
from PIL import Image
import sys
import argparse
import subprocess as sp

# Chip properties
CHIP_SIZE = 256
STRIDE    = 256
WATER_MIN = 256*256//4
WATER_MAX = 256*256 - WATER_MIN

# Multiprocessing config
N_PROC    = 8

# Set by argparse
WORK_DIR  = None #temp for s2,dw -- fast,local
CHIP_DIR  = None #output dir -- fast,local
S2_DIR    = None #large storage -- slow
DW_DIR    = None #large storage -- slow

class Product():
	def __init__(self,safe_path,label_path):
		# Absolute paths
		self.safe_path  = safe_path #dir
		self.label_path = label_path #single file

		# Strings
		self.safe_id = safe_path.split('/')[-1]
		self.base_id = self.set_chip_base_str()

		# rasterio.DatasetReader
		self.s2_readers = [rio.open(p,'r',tiled=True) for p in self.get_band_paths()]
		self.dw_reader  = rio.open(self.label_path,'r',tiled=True)

		# ALIGN RASTERS
		self.s2_borders,self.dw_borders = self.align_s2_dw(self.s2_readers[0],self.dw_reader)


	def set_chip_base_str(self):
			orbit = self.safe_id.split('_')[4]
			return f"{self.label_path.split('/')[-1].rstrip('.tif')}_{orbit}"


	def get_band_paths(self):
		date = self.safe_id.split('_')[2]
		y = date[0:4]
		m = date[4:6]
		d = date[6:8]
		band_regex = f"GRANULE/*/IMG_DATA/R10m/*_10m.jp2"
		paths = [f"{self.safe_path}/{s}" for s in glob.glob(band_regex,root_dir=self.safe_path)]
		return paths


	def align_s2_dw(self,s2_src:rio.DatasetReader,dw_src:rio.DatasetReader) -> Tuple:
		'''
		Match indices and remove borders.
		'''
		########## 1. REMOVE DW NO-DATA BORDERS(~1-2px each side) ##########
		dw_top    = 0
		dw_bottom = dw_src.height-1
		dw_left   = 0
		dw_right  = dw_src.width-1

		while(True):
			row = dw_src.read(1,window=rio.windows.Window(0,dw_top,dw_src.width,1))
			if row.sum() == 0:
				dw_top += 1
			else:
				break

		while(True):
			row = dw_src.read(1,window=rio.windows.Window(0,dw_bottom,dw_src.width,1))
			if row.sum() == 0:
				dw_bottom -= 1
			else:
				break

		while(True):
			col = dw_src.read(1,window=rio.windows.Window(dw_left,0,1,dw_src.height))
			if col.sum() == 0:
				dw_left += 1
			else:
				break

		while(True):
			col = dw_src.read(1,window=rio.windows.Window(dw_right,0,1,dw_src.height))
			if col.sum() == 0:
				dw_right -= 1
			else:
				break

		dw_ij = {'top':dw_top, 'bottom':dw_bottom, 'left':dw_left, 'right':dw_right}

		########### 2.MATCH DW to S2 (DW has ~20px less on each side) ##########
		# DW ij's (px indices) to DW xy's (coords)
		dw_xy_ul = dw_src.xy(dw_ij['top'],dw_ij['left'],offset='center')
		dw_xy_lr = dw_src.xy(dw_ij['bottom'],dw_ij['right'],offset='center')

		# DW xy's (coords) to S2 ij's (px indices) -- DW is smaller
		s2_ij = {}
		s2_ij['top'],s2_ij['left']     = s2_src.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
		s2_ij['bottom'],s2_ij['right'] = s2_src.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

		########## 3.REMOVE S2 TILE OVERLAP & ADJUST DW ACCORDINGLY ##########
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


def get_datastrip_id(str):
	'''
	Given a string like /*.SAFE/GRANULE/SUBFOLDER, use SUBFOLDER to return
	datastrip id
	'''
	pass


def get_dw_id(safe_str,datastrip_str):
	'''
	Build DynamicWorldV1 (DW) product ID from *.SAFE folder name and datastrip str.
	DW follows {date}_{datastrip}_{tile} convention.
	'''
	# tile  = safe_str[38:44]
	# date  = safe_str[11:26]
	tile = safe_str.split('_')[-2]
	date = safe_str.split('_')[2] #more readable
	return f"{date}_{datastrip_str}_{tile}.tif"


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


def chip_image(product):

	# STDOUT
	start_time = time.perf_counter()

	# LOAD BAND ARRAYS, CLIP, & NORMALIZE
	rgbn  = []
	zeros = []
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
		low_cutoff  = 0
		band_array  = np.clip(band_array,low_cutoff,high_cutoff)
		band_array  = np.round((band_array-low_cutoff)/(high_cutoff-low_cutoff)*255)
		band_array  = band_array.astype(np.uint8)
		zeros.append(zero_mask)
		rgbn.append(band_array)

	# SET WINDOWS
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
			args=(rgbn,zeros,product.label_path,s2_window_chunks[i],dw_window_chunks[i],product.base_id,lock)
		)
		p.start()
		processes.append(p)

	for i,p in enumerate(processes):
		p.join(timeout=60)
		if p.is_alive():
			print(f"Worker {i} timed out.")
		elif p.exitcode != 0:
			print(f"Worker {i} exited with code {p.exitcode}")

	# STDOUT	
	exec_time = time.perf_counter() - start_time
	print(f"All workers done ({exec_time:.3f} secs). ")


def chip_image_worker(rgbn,zeros,label_path,s2_windows,dw_windows,base_id,lock):

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
		zero_window = zeros[w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
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
		lbl_arr[lbl_arr!=1] = 0
		lbl_arr[lbl_arr==1] = 255 #set positive to 255, negative to 0
		Image.fromarray(lbl_array).save(outfile)

		# STATS/LOG
		stats.append(f'{outfile.split('/')[-1]}\t{n_water}')

	# LOG
	lock.acquire()
	with open(f'{CHIP_DIR}/stats.txt','a') as fp:
		fp.write('\n'.join(stats))
	lock.release()


def parse_args():

	########## ARGV CONFIG ##########
	parser = argparse.ArgumentParser(
		prog="chip.py",
		description="Sentinel-2 and DynamicWorld V1 Products to 256x256 images.")
	parser.add_argument('--work-dir',default=None,
		help="Temporary directory to load/offload data.")
	parser.add_argument('--chip-dir',default=None,
		help="Output directory for resulting chips")
	parser.add_argument('--s2-dir',default=None,
		help="Source directory for raw Sentinel-2 products.")
	parser.add_argument('--dw-dir',default=None,
		help="Source directory for 10980x10980 mask rasters.")

	########## SET ARGS ##########
	args = parser.parse_args()

	global WORK_DIR
	global CHIP_DIR
	global S2_DIR
	global DW_DIR

	WORK_DIR = args.work_dir
	CHIP_DIR = args.chip_dir
	S2_DIR   = args.s2_dir 
	DW_DIR   = args.label_dir

	if not os.path.isdir(WORK_DIR):
		print(f"WORK_DIR {WORK_DIR} not found. EXIT(1).")
		sys.exit(1)
	if WORK_DIR[-1] == '/':
		WORK_DIR = WORK_DIR.rstrip('/')

	if CHIP_DIR is None:
		os.makedirs(WORK_DIR + '/chips',exist_ok=True)
		CHIP_DIR = WORK_DIR + '/chips'
	if not os.path.isdir(CHIP_DIR):
		print(f"CHIP_DIR in {CHIP_DIR} not found. EXIT(1).")
		sys.exit(1)

	if not os.path.isdir(S2_DIR):
		print("S2_DIR not found. EXIT(1).")
		sys.exit(1)
	if S2_DIR[-1] == '/':
		S2_DIR = S2_DIR.rstrip('/')

	if not os.path.isdir(LABEL_DIR):
		print("LABEL_DIR not found. EXIT(1).")
		sys.exit(1)
	if LABEL_DIR[-1] == '/':
		LABEL_DIR = LABEL_DIR.rstrip('/')

	print(f"WORK_DIR set to: {WORK_DIR}")
	print(f"CHIP_DIR set to: {CHIP_DIR}")
	print(f"S2_DIR set to:   {S2_DIR}")
	print(f"DW_DIR set to:   {DW_DIR}")	


if __name__ == '__main__':
	########## ARGS ##################################	
	parse_args()

	########## FILTER ################################ #only products matching label
	safe_regex   = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE"
	remote_safes = glob.glob(safe_regex,root_dir=S2_DIR) #returns ['/eodata/.../*.SAFE']

	subdirs    = [glob.glob("*",root_dir=f"{s}/GRANULE")[0] for s in safes]
	datastrips = [s.split('_')[-1] for s in subdirs]
	dw_ids     = [get_gee_id(s,d) for s,d in zip(remote_safes,datastrips)]	

	dw_files     = glob.glob("*.tif",root_dir=DW_DIR) # actual label dir
	intersection = np.isin(dw_ids,dw_files)
	good_safes   = remote_safes[intersection]
	good_labels  = dw_ids[intersection] #match order


	########## SPLIT AND QUEUE ################
	chunk_size  = 50
	N_chunks    = len(good_safes) // chunk_size
	remainder   = len(good_safes) % chunk_size
	chunk_queue = [] # [[(s2_str,dw_str)]]
	for i in range(N_chunks):
		chunk_safes  = good_safes[i*chunk_size:i*chunk_size+chunk_size]
		chunk_labels = good_labels[i*chunk_size:i*chunk_size+chunk_size]
		chunk_queue.append(list(zip(chunk_safes,chunk_labels)))
	if remainder != 0:
		chunk_safes  = good_safes[N_chunks*chunk_size:]
		chunk_labels = good_labels[N_chunks*chunk_size:]
		chunk_queue.append(list(zip(chunk_safes,chunk_labels)))


	########## BATCH PROCESS #########################
	for chunk in chunk_queue:

		########## COPY CHUNK DATA #######################
		for safe_path,label_path in chunk:
			sp.run(["cp","-v","-r",f"{S2_DIR}/{safe_path}",WORK_DIR]) # COPY BANDS
			sp.run(["cp","-v",f"{DW_DIR}/{label_path}",WORK_DIR]) # COPY LABEL

		########## CHIP ##################################
		N = len(chunk)
		for i,(safe_path,label_path) in enumerate(chunk):
			safe_local_path  = f"{WORK_DIR}/{safe_path.split("/")[-1]}" #remove 'eodata/../'
			label_local_path = f"{WORK_DIR}/{label_path}"
			product = Product(safe_local_path,label_local_path)
			print(f'[{i+1}/{N}] PROCESSING {product.base_id}')		
			chip_image(product)
