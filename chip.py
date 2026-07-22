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
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Chip properties
CHIP_SIZE = 256
STRIDE    = 256
WATER_MIN = 256*256//4
WATER_MAX = 256*256 - WATER_MIN

# Default raster properties
RASTER_SIZE = 10980
OVERLAP_MIN = 492
OVERLAP_MAX = RASTER_SIZE - 492

# Multiprocessing settings
N_PROC = 8
N_COPY = 5

# Globals set by argparse
WORK_DIR  = None #temp for s2,dw -- fast,local
CHIP_DIR  = None #output folder -- fast,local
S2_DIR    = None #large storage -- slow
DW_DIR    = None #large storage -- slow

class EmptyLabelError(Exception):
	'''
	Empty DynamicWorld raster.
	'''
	pass



class Product():
	'''
	Class to keep paths, image readers, and usable area of rasters.
	1 Product corresponds to a unique image-label pair (Sentinel-2 bands and
	DynamicWorld image)
	'''
	def __init__(self,safe_path,label_path):
		# Absolute paths
		self.safe_path  = safe_path #dir
		self.label_path = label_path #single file

		# Strings
		self.safe_id = safe_path.split('/')[-1]
		self.base_id = self.set_chip_base_str()

		# rasterio.DatasetReader for bands and labels
		self.s2_readers = [rio.open(p,'r',tiled=True) for p in self.get_band_paths()]
		self.dw_reader  = rio.open(self.label_path,'r',tiled=True)

		# Check empty labels
		if self.dw_reader.statistics(1).max == 0:
			raise EmptyLabelError("Label is zero everywhere.")

		# ALIGN RASTERS
		self.s2_borders,self.dw_borders = self.align_s2_dw(self.s2_readers[0],self.dw_reader)


	def set_chip_base_str(self):
		'''
		Unique base name for all chips extracted from a product.
		Set to:
			{date}_{datastrip}_{tile}_{orbit}

		When chipping (i.e. chip_image() is called), chip ids are set to:
			{date}_{datastrip}_{tile}_{orbit}_{row}_{col}_{band or label}.tif
			
		'''
		orbit = self.safe_id.split('_')[4]
		return f"{self.label_path.split('/')[-1][:-4]}_{orbit}"


	def get_band_paths(self):
		'''
		Get paths for individual .jp2 band files within *.SAFE folder
		'''
		band_regex = f"{self.safe_path}/GRANULE/*/IMG_DATA/R10m/*_B0[2348]_10m.jp2"
		paths = sorted(glob.glob(band_regex)) # returns B02,B03,B04,B08 -- BGRN 
		paths = paths[::-1][1:] + [paths[-1]] # set to RGBN - 04,03,02,08
		return paths


	def align_s2_dw(self,s2_src,dw_src):
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

		########## 3.REMOVE S2 TILE OVERLAP & MATCH DW ##########
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

		# Return two dicts
		return s2_ij,dw_ij	


	def close(self):
		'''
		Close all DatasetReader's
		'''
		for reader in self.s2_readers:
			if not reader.closed:
				reader.close()
		if not self.dw_reader.closed:
			self.dw_reader.close()



def get_datastrip_id(safe_path):
	'''
	Given a string like /*.SAFE/GRANULE/SUBFOLDER, use SUBFOLDER to return
	datastrip id. Takes absolute paths.
	'''
	subdir = glob.glob("*",root_dir=f"{safe_path}/GRANULE")[0]
	return subdir.split('_')[-1]



def get_dw_id(safe_str,datastrip_str):
	'''
	Build DynamicWorldV1 (DW) product ID from *.SAFE folder name and datastrip str.
	DW follows {date}_{datastrip}_{tile} convention.
	Takes absolute paths.
	'''
	# tile  = safe_str[38:44]
	# date  = safe_str[11:26]
	tile = safe_str.split('/')[-1].split('_')[-2]
	date = safe_str.split('/')[-1].split('_')[2] #more readable
	return f"{date}_{datastrip_str}_{tile}.tif"



def get_strided_windows(borders):
	'''
	Given a dicts of boundaries, returns an array list with tuples (i,j) for block indices i,j and 
	window objects corresponding to the block i,j. This considers only the usable area of the raster,
	defined by the dict of borders. For an array with two rows and two columns of no data (zeros), 
	the blocks are offseted and defined as:

			    left   stride   stride*2
				| 0 0...  	      |
				| 0 0... |		  | 
	top      ---+--------+--------+----
		    0 0 |        |        |
		    0 0 | (0, 0) | (0, 1) | ...
		     .  |        |        |
	stride*1 .  +--------+--------+
		     .  |        |        |
		        | (1, 0) | (1, 1) | ...
		        |        |        |
	stride*2 ---+--------+--------+---
				|  ...   |   ...  |


	Parameters
	----------
	borders: dict
		The dictionary containing the first and last indices of usable data.

	Returns
	-------
	List of [(str,str),Window]. Window object is arg to .read() method in 
	rasterio.DatasetReader. Indices i,j are row,col position of chip in the 
	original raster.
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
		windows += [[(i,j),W]]

	return windows



def chip_image(product):
	'''
	Process a Sentinel-2 product and its corresponding DynamicWorld V1 product.
	This is done through these steps:
	1. Clip rgb-nir bands at 99th percentile, set to 255.
	2. Divide the arrays into windows
	3. Divide windows into to multiple sections
	4. Assign each section to a separate process.

	Each process calls chip_image_worker().
	'''

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

		if high_cutoff <= low_cutoff: #avoid division by zero
			print(f"Degenerate band range in {reader.files[0]} -- SKIPPING.")
			return

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
			p.terminate()
			p.join()
		elif p.exitcode != 0:
			print(f"Worker {i} exited with code {p.exitcode}")

	# STDOUT	
	exec_time = time.perf_counter() - start_time
	print(f"All workers done ({exec_time:.3f} secs). ")



def chip_image_worker(rgbn,zeros,label_path,s2_windows,dw_windows,base_id,lock):
	'''
	Function for worker processes.
	Checks both no-data and water-land ratio.
	Stores 4 bands as *_B0X.tif file, where alpha channel is NIR band (band 8).
	Stores label as 0 or 255 values for land and water respectively.
	Number of water pixels per chip is written to stats.txt.
	'''
	# Label DatasetReader -- per thread
	lbl_rdr = rio.open(label_path,'r',tiled=True)

	# Single zero mask
	zero_stack = np.any(zeros,axis=0)

	# CHIP INFO
	stats = []

	for k,(rowcol,w) in enumerate(s2_windows):

		# LOAD (WINDOW SECTION) LABEL
		lbl_array = lbl_rdr.read(1,window=dw_windows[k][1])

		# CHECK LABEL NO DATA -- SKIP CHIP 
		if (lbl_array == 0).any():
			continue

		# CHECK WATER/LAND RATIO
		n_water = (lbl_array==1).sum()
		if n_water < WATER_MIN or n_water > WATER_MAX:
			continue

		# CHECK RGB-NIR NO DATA
		zero_window = zero_stack[w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		if zero_window.sum() > 0:
			continue

		# LOAD RGB
		r_array = rgbn[0][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		g_array = rgbn[1][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		b_array = rgbn[2][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]
		n_array = rgbn[3][w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE]

		# SAVE BANDS
		row,col = rowcol
		outfile = f"{CHIP_DIR}/{base_id}_{row:02}_{col:02}_B0X.tif"
		r = Image.fromarray(r_array)
		g = Image.fromarray(g_array)
		b = Image.fromarray(b_array)
		n = Image.fromarray(n_array)
		Image.merge('RGBA',(r,g,b,n)).save(outfile)

		# SAVE LABEL
		outfile = f"{CHIP_DIR}/{base_id}_{row:02}_{col:02}_LBL.tif"
		lbl_array[lbl_array!=1] = 0
		lbl_array[lbl_array==1] = 255 #set positive to 255, negative to 0
		Image.fromarray(lbl_array).save(outfile)

		# STATS/LOG
		stats.append(f"{outfile.split('/')[-1]}\t{n_water}")

	# LOG
	if len(stats) > 0:
		lock.acquire()
		with open(f'{CHIP_DIR}/stats.txt','a') as fp:
			fp.write('\n'.join(stats) + '\n')
		lock.release()

	# CLEAR
	lbl_rdr.close()



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
	DW_DIR   = args.dw_dir

	if not os.path.isdir(WORK_DIR):
		print(f"WORK_DIR {WORK_DIR} not found. EXIT(1).")
		sys.exit(1)
	if WORK_DIR[-1] == '/':
		WORK_DIR = WORK_DIR.rstrip('/')

	if CHIP_DIR is None:
		os.makedirs(f"{WORK_DIR}/chips_{CHIP_SIZE}",exist_ok=True)
		CHIP_DIR = f"{WORK_DIR}/chips_{CHIP_SIZE}"
	if not os.path.isdir(CHIP_DIR):
		print(f"CHIP_DIR in {CHIP_DIR} not found. EXIT(1).")
		sys.exit(1)

	if not os.path.isdir(S2_DIR):
		print("S2_DIR not found. EXIT(1).")
		sys.exit(1)
	if S2_DIR[-1] == '/':
		S2_DIR = S2_DIR.rstrip('/')

	if not os.path.isdir(DW_DIR):
		print("LABEL_DIR not found. EXIT(1).")
		sys.exit(1)
	if DW_DIR[-1] == '/':
		DW_DIR = DW_DIR.rstrip('/')

	print(f"WORK_DIR set to: {WORK_DIR}")
	print(f"CHIP_DIR set to: {CHIP_DIR}")
	print(f"S2_DIR set to:   {S2_DIR}")
	print(f"DW_DIR set to:   {DW_DIR}")	


def copy_pair(safe_path,label_path):
	'''
	Worker function for multi-threaded copy in loop.
	'''
	dst = os.path.join(WORK_DIR,os.path.basename(safe_path))
	shutil.copytree(f"{S2_DIR}/{safe_path}",dst,dirs_exist_ok=True)
	shutil.copy2(f"{DW_DIR}/{label_path}",WORK_DIR)


if __name__ == '__main__':
	########## ARGS ##################################	
	parse_args()

	########## FILTER ################################ #only products matching label
	safe_regex   = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE"
	remote_safes = glob.glob(safe_regex,root_dir=S2_DIR) #returns ['/eodata/.../*.SAFE']

	datastrips = [get_datastrip_id(f"{S2_DIR}/{s}") for s in remote_safes]
	dw_ids     = [get_dw_id(f"{S2_DIR}/{s}",d) for s,d in zip(remote_safes,datastrips)]

	dw_files     = glob.glob("*.tif",root_dir=DW_DIR) # actual label dir
	intersection = np.isin(dw_ids,dw_files)
	good_safes   = np.array(remote_safes)[intersection].tolist() #1035
	good_labels  = np.array(dw_ids)[intersection].tolist() #match order


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
	for j,chunk in enumerate(chunk_queue):
		print(f"[BATCH {j+1}/{len(chunk_queue)}]",flush=True)
		N = len(chunk)

		########## COPY DATA IN CHUNK ####################
		# for i,(safe_path,label_path) in enumerate(chunk):
			# sp.run(["cp","-r",f"{S2_DIR}/{safe_path}",WORK_DIR]) # COPY BANDS (flat on dir)
			# sp.run(["cp",f"{DW_DIR}/{label_path}",WORK_DIR]) # COPY LABEL
			# if (i+1) % 10 == 0 or (i+1) == N:
				# print(f"Copied {i+1}/{N} image-label pairs")
		with ThreadPoolExecutor(max_workers=N_COPY) as executor:
			futures = [executor.submit(copy_pair,sp,lp) for sp,lp in chunk]
			for i,future in enumerate(as_completed(futures)):
				future.result()
				if (i+1) % 10 == 0 or (i+1) == N:
					print(f"Copied {i+1}/{N} image-label pairs",flush=True)

		########## CHIP ##################################
		for i,(safe_path,label_path) in enumerate(chunk):
			safe_local_path  = f"{WORK_DIR}/{safe_path.split('/')[-1]}" #remove 'eodata/../'
			label_local_path = f"{WORK_DIR}/{label_path}"

			# LOAD IMAGE PATHS AND READERS
			try:
				product = Product(safe_local_path,label_local_path)
			except Exception as e:
				print(f"Error in Product.__init__() for {safe_local_path}.")
				print(f"Error: {e}")
				continue

			# CHIP
			try:
				print(f'[{i+1}/{N}] PROCESSING {product.base_id}')	
				chip_image(product)
			finally:
				product.close()

		########## DELETE CHUNK DATA #####################
		for safe_path,label_path in chunk:
			safe_local_path  = f"{WORK_DIR}/{safe_path.split('/')[-1]}" #remove 'eodata/../'
			label_local_path = f"{WORK_DIR}/{label_path}"
			print(f"Deleting {safe_local_path}")
			shutil.rmtree(safe_local_path)	
			print(f"Deleting {label_local_path}")
			os.remove(label_local_path)
