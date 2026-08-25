'''
Several plots/figures describing the dataset.
'''
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import os
from PIL import Image
from glob import glob
import math
import argparse
from datetime import datetime
from shapely import wkt

# HARD-CODED US STATE POLYGONS SOURCE
STATE_SHP = "../cb_2024_us_state_500k/cb_2024_us_state_500k.shp"
PLOT_CRS  = "EPSG:5070"

# DYNAMICWORLD COLORS & LABELS
dw_colors = {
0:"419bdf",
1:"397d49",
2:"88b053",
3:"7a87c6",
4:"e49635",
5:"dfc35a",
6:"c4281b",
7:"a59b8f",
8:"b39fe1"}

dw_labels = {
0:"water",
1:"trees",
2:"grass",
3:"flooded_vegetation",
4:"crops",
5:"shrub_and_scrub",
6:"built",
7:"bare",
8:"snow_and_ice"}

# MATPLOTLIB PRESETS
plt.style.use('fast')
plt.rcParams['font.family'] = 'Courier'
plt.rcParams['font.size'] = 10
fig_size_normal=(8,5)
fig_size_wide=(16,4)

################################################################################
# POLYGONS
################################################################################
def filter_tiles(geo_df,drop):
	'''
	Parameters:
	-----------
	geo_df: gpd.GeoDataFrame
	drop: [str]
	'''
	pass


def plot_tile_polygons():
	'''
	Plot a combined set of polygons for AOI and MGRS tiles.

	Parameters:
	----------
	us_geom: str
		path to a .shp file for US state geometries.
	'''
	# ---------------------------------------------------------------------------
	# 1. LOAD FILES
	# ---------------------------------------------------------------------------	
	# LOAD US GEOM
	states      = gpd.read_file(STATE_SHP)
	territories = ['PR','AS','VI','MP','GU','AK','HI']
	contiguous  = states[~states['STUSPS'].isin(territories)]

	# TILES IN DATASET
	tile_geometries_file = "./search/search_results_geometries.tsv"
	with open(tile_geometries_file,'r') as fp:
		lines = fp.readlines()
	safe_ids  = [l.split('\t')[0] for l in lines]
	tile_geom = [l.split('\t')[1].split(';')[1].rstrip("'") for l in lines]

	# ---------------------------------------------------------------------------
	# 2. EXTRACT MGRS TILE ID/SEPARATE DROPPED
	# ---------------------------------------------------------------------------
	mgrs = [s.split("_")[5] for s in safe_ids]
	unique_mgrs,unique_mgrs_idx = np.unique(mgrs,return_index=True)
	unique_tile_geom = np.array(tile_geom)[unique_mgrs_idx]

	dropped_tiles = ["T11SKD","T11TKE"]

	bad_mask = np.isin(unique_mgrs,dropped_tiles)
	bad_mgrs      = unique_mgrs[bad_mask]
	bad_tile_geom = unique_tile_geom[bad_mask]
	good_mgrs      = unique_mgrs[~bad_mask]
	good_tile_geom = unique_tile_geom[~bad_mask]

	# ---------------------------------------------------------------------------
	# 3. PARSE WKT GEOMETRIES
	#    Raw format: geography'SRID=4326;POLYGON ((...))' — strip the prefix.
	# ---------------------------------------------------------------------------
	good_tile_wkts = [wkt.loads(s) for s in good_tile_geom]
	tile_df = pd.DataFrame({"tile": good_mgrs})
	tile_gdf = gpd.GeoDataFrame(
	    tile_df,
	    geometry=good_tile_wkts,
	    crs="EPSG:4326"
	)

	bad_tile_wkts = [wkt.loads(s) for s in bad_tile_geom]
	bad_tile_df   = pd.DataFrame({"tile": bad_mgrs})
	bad_tile_gdf  = gpd.GeoDataFrame(
	    bad_tile_df,
	    geometry=bad_tile_wkts,
	    crs="EPSG:4326"
	)

	# ---------------------------------------------------------------------------
	# 4. PROJECT TO COMMON CRS
	# ---------------------------------------------------------------------------
	contiguous    = contiguous.to_crs(PLOT_CRS)
	tile_gdf      = tile_gdf.to_crs(PLOT_CRS)
	bad_tile_gdf  = bad_tile_gdf.to_crs(PLOT_CRS)

	# ---------------------------------------------------------------------------
	# 5. PLOT
	# ---------------------------------------------------------------------------
	fig, ax = plt.subplots(1,1,figsize=(12,12))

	contiguous.plot(ax=ax,color='white',alpha=1.0,edgecolor='black',linewidth=0.25)
	tile_gdf.plot(ax=ax,color='blue',alpha=0.2,edgecolor='blue',linewidth=1.0)
	bad_tile_gdf.plot(ax=ax,color='red',alpha=0.1,edgecolor='red',linewidth=1.0)

	# zoom in
	xmin, ymin, xmax, ymax = contiguous.total_bounds
	# print(f"X:{xmin}--{xmax} | Y: {ymin}--{ymax}")
	x_range = xmax - xmin
	y_range = ymax - ymin
	xmax = x_range*0.3 + xmin
	ymax = y_range*0.85 + ymin	
	ymin = y_range*0.25 + ymin


	ax.set_xlim(xmin,xmax)
	ax.set_ylim(ymin,ymax)

	# title, layout, etc
	# ax.set_title("Sentinel-2 Tiles",fontsize=24)
	ax.set_axis_off()
	plt.tight_layout()
	plt.savefig("./figures/tiles.png")


def plot_tile_polygons_with_split():

	# with open('./split.txt','r') as fp:
	# 	lines = fp.readlines()
	# ...etc etc

	te_tiles = ['T11SQV', 'T10TEL', 'T10SFG']
	va_tiles = ['T10TFK', 'T11SLC', 'T10SEH']
	tr_tiles = ['T10TFL', 'T12TVK', 'T11SQA', 'T10SEJ', 'T10SGJ', 'T11SPS', 'T12TXL', 'T12TUL', 'T10TFM', 'T11SQU', 'T12TVM', 'T10TGK', 'T12SVG']
	tile_geometries_file = "./search/search_results_geometries.tsv"

	# ---------------------------------------------------------------------------
	# 1. LOAD FILES
	# ---------------------------------------------------------------------------	
	# LOAD US GEOM
	states      = gpd.read_file(STATE_SHP)
	territories = ['PR','AS','VI','MP','GU','AK','HI']
	contiguous  = states[~states['STUSPS'].isin(territories)]

	# TILES IN DATASET
	tile_geometries_file = "./search/search_results_geometries.tsv"
	with open(tile_geometries_file,'r') as fp:
		lines = fp.readlines()
	safe_ids  = [l.split('\t')[0] for l in lines]
	tile_geom = [l.split('\t')[1].split(';')[1].rstrip("'") for l in lines]

	# ---------------------------------------------------------------------------
	# 2. EXTRACT MGRS TILE ID/SEPARATE DROPPED
	# ---------------------------------------------------------------------------
	mgrs = [s.split("_")[5] for s in safe_ids]
	unique_mgrs,unique_mgrs_idx = np.unique(mgrs,return_index=True)
	unique_tile_geom = np.array(tile_geom)[unique_mgrs_idx]

	dropped_tiles = ["T11SKD","T11TKE"]

	bad_mask = np.isin(unique_mgrs,dropped_tiles)
	bad_mgrs      = unique_mgrs[bad_mask]
	bad_tile_geom = unique_tile_geom[bad_mask]
	good_mgrs      = unique_mgrs[~bad_mask]
	good_tile_geom = unique_tile_geom[~bad_mask]

	te_mask = np.isin(good_mgrs,te_tiles)
	va_mask = np.isin(good_mgrs,va_tiles)
	tr_mask = np.isin(good_mgrs,tr_tiles)

	te_mgrs = good_mgrs[te_mask]
	va_mgrs = good_mgrs[va_mask]
	tr_mgrs = good_mgrs[tr_mask]
	te_geom = good_tile_geom[te_mask]
	va_geom = good_tile_geom[va_mask]
	tr_geom = good_tile_geom[tr_mask]

	# ---------------------------------------------------------------------------
	# 3. PARSE WKT GEOMETRIES
	# ---------------------------------------------------------------------------
	te_wkts = [wkt.loads(s) for s in te_geom]
	te_df = pd.DataFrame({"tile": te_mgrs})
	te_gdf = gpd.GeoDataFrame(
	    te_df,
	    geometry=te_wkts,
	    crs="EPSG:4326"
	)

	va_wkts = [wkt.loads(s) for s in va_geom]
	va_df = pd.DataFrame({"tile": va_mgrs})
	va_gdf = gpd.GeoDataFrame(
	    va_df,
	    geometry=va_wkts,
	    crs="EPSG:4326"
	)

	tr_wkts = [wkt.loads(s) for s in tr_geom]
	tr_df = pd.DataFrame({"tile": tr_mgrs})
	tr_gdf = gpd.GeoDataFrame(
	    tr_df,
	    geometry=tr_wkts,
	    crs="EPSG:4326"
	)

	# ---------------------------------------------------------------------------
	# 4. PROJECT TO COMMON CRS
	# ---------------------------------------------------------------------------
	contiguous    = contiguous.to_crs(PLOT_CRS)
	te_gdf      = te_gdf.to_crs(PLOT_CRS)
	va_gdf      = va_gdf.to_crs(PLOT_CRS)
	tr_gdf      = tr_gdf.to_crs(PLOT_CRS)

	# ---------------------------------------------------------------------------
	# 5. PLOT
	# ---------------------------------------------------------------------------
	fig, ax = plt.subplots(1,1,figsize=(12,12))

	contiguous.plot(ax=ax,color='white',alpha=1.0,edgecolor='black',linewidth=0.25)
	te_gdf.plot(ax=ax,color='blue',alpha=0.2,edgecolor='blue',linewidth=1.0)
	va_gdf.plot(ax=ax,color='red',alpha=0.2,edgecolor='red',linewidth=1.0)
	tr_gdf.plot(ax=ax,color='green',alpha=0.2,edgecolor='red',linewidth=1.0)

	# zoom in
	xmin, ymin, xmax, ymax = contiguous.total_bounds
	# print(f"X:{xmin}--{xmax} | Y: {ymin}--{ymax}")
	x_range = xmax - xmin
	y_range = ymax - ymin
	xmax = x_range*0.3 + xmin
	ymax = y_range*0.85 + ymin	
	ymin = y_range*0.25 + ymin
	ax.set_xlim(xmin,xmax)
	ax.set_ylim(ymin,ymax)

	# title, layout, etc
	# ax.set_title("Sentinel-2 Tiles",fontsize=24)
	ax.set_axis_off()
	plt.tight_layout()
	plt.savefig("./figures/tiles_split.png")


################################################################################
# ENTIRE TILE RASTERS
################################################################################
def plot_s2(band_folder):
	'''
	Plot an RGB image.
	'''

	assert os.path.isdir(band_folder), f"No directory found in {band_folder}"

	# GET PATHS AND RASTERIO READERS/POINTERS
	band_regex = f"{band_folder}/GRANULE/*/IMG_DATA/R10m/*_B0[2348]_10m.jp2"
	band_paths = sorted(glob.glob(band_regex))
	# band_paths = band_paths[::-1][1:] + [band_paths[-1]]
	band_paths = band_paths[::-1][1:] # RGB
	band_readers = [rio.open(p,'r',tiled=True) for p in band_paths]

	# BAND METADATA
	band_kwargs = band_readers[0].meta.copy()
	band_kwargs.update({
		'count':3,
		'driver':'JP2OpenJPEG','codec':'J2K',
		'dtype':'uint8',
		'quality':100,
		'reversible':True
		})


	#LOAD BANDS
	r,g,b = (_.read(1) for _ in band_readers[0:3])
	rgb   = np.stack([r,g,b],axis=0).astype(np.uint16)

	# GET ZEROS
	zero_mask = rgb == 0
	and_zero_mask = zero_mask.all(axis=0)

	# CLIP AND NORMALIZE -- FOR PLOT
	# max_cutoff = [np.percentile(b[~rgb_zero_mask[i]],99) for i,b in enumerate(rgb)]
	max_cutoff = np.percentile(rgb[~zero_mask],99)
	rgb = np.clip(rgb,0,max_cutoff)
	rgb = np.round(rgb/max_cutoff * 255).astype(np.uint8)

	# SET ZEROS
	rgb[:,and_zero_mask] = [[0],[0],[0]]

	# SET OUT PATH
	safe_id  = img_path.split('/')[-1].split(".SAFE")[0]
	out_path = f"./figures/{safe_id}_bands.tif"

	# OUT PTR WRITE
	out_ptr.open(out_path,'w',**kwargs)
	out_ptr.write(rgb,indexes=[1,2,3])
	out_ptr.close()

	# STDOUT
	print(f"file written to {out_path}")


def plot_dw(img_path):
	'''
	Plot a DynamicWorld V1 image.
	'''

	assert os.path.isfile(img_path), f"No file in {img_path}"

	# READ METADATA
	in_ptr = rio.open(img_path,'r')

	# READ ARRAY
	dw_arr = in_ptr.read(1)

	# FIND CATEGORIES
	white_mask = dw_arr == 1 #water
	gray_mask  = dw_arr == 0 #no data
	black_mask = ~(white_mask | gray_mask) #land/everything else

	# SET COLORS
	dw_arr[white_mask] = 255
	dw_arr[black_mask] = 0
	dw_arr[gray_mask]  = 128
	arr_3d = np.repeat(dw_arr[np.newaxis,:,:],repeats=3,axis=0)

	# SET OUT PATH
	dw_tile_id = img_path.split('/')[-1].split('.')[0]
	out_path   = f"./figures/{dw_tile_id}_binary.tif"	

	# OUT POINTER WRITE
	meta_kwargs = in_ptr.meta.copy()	
	meta_kwargs.update({'count':3,'compress':'lzw'})
	out_ptr = rio.open(out_path,'w',**meta_kwargs)
	out_ptr.write(arr_3d)
	out_ptr.close()
	print(f"file written to {out_path}")


def plot_dw_original(img_path):
	'''
	Plot a DynamicWorld V1 image. All 8 original classes.
	'''

	assert os.path.isfile(img_path), f"No file in {img_path}"

	# READ METADATA
	in_ptr = rio.open(img_path,'r')
	meta_kwargs = in_ptr.meta.copy()

	# READ ARRAY
	dw_arr = in_ptr.read(1)

	# FINAL ARRAY
	out_arr = np.zeros((3,dw_arr.shape[0],dw_arr.shape[1])).astype(np.uint8)

	for original_label in dw_colors:
		hex_value = dw_colors[original_label]
		r = int(hex_value[0:2], 16)
		g = int(hex_value[2:4], 16)
		b = int(hex_value[4:6], 16)
		mask = dw_arr == original_label+1 #stored images are shifted

		out_arr[:,mask] = [[r],[g],[b]]

	# SET NO-DATA AS BLACK
	no_data_mask = dw_arr == 0
	out_arr[:,no_data_mask] = [[0],[0],[0]]

	# SET OUT PATH
	dw_tile_id = img_path.split('/')[-1].split('.')[0]
	out_path = f"./figures/{dw_tile_id}_color.tif"

	# POINTER WRITE
	meta_kwargs = in_ptr.meta.copy()	
	meta_kwargs.update({'count':3,'compress':'lzw'})
	out_ptr = rio.open(out_path,'w',**meta_kwargs)
	out_ptr.write(out_arr)
	out_ptr.close()
	print(f"file written to {out_path}")


################################################################################
# ENTIRE TILE RASTERS -- GRIDS
################################################################################
def get_borders(s2_reader,dw_reader):
	'''
	Align and get usable are as done in chipping (chip.py)
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


def get_windows(borders):
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

	
def plot_bands_windows(img_path):
	pass


def plot_label_windows(s2_img_path,dw_img_path,chip_size=256,stride=256):

	assert os.path.isfile(dw_img_path), f"No file in {dw_img_path}"
	assert os.path.isfile(s2_img_path), f"No file in {s2_img_path}"

	# INPUT READERS
	s2_reader = rio.open(s2_img_path,'r',tiled=True)
	dw_reader = rio.open(dw_img_path,'r',tiled=True)

	# COPY METADATA AND SET OUTPUT PTR
	meta_kwargs = dw_reader.meta.copy()	
	meta_kwargs.update({'count':3,'compress':'lzw','height':s2_reader.height,'width':s2_reader.width})
	out_ptr = rio.open(out_path,'w',**meta_kwargs)

	# FIND ALIGNED BOUNDARIES
	s2_borders,dw_borders = get_borders(s2_reader,dw_reader)
	s2_windows = get_windows(s2_borders)
	dw_windows = get_windows(dw_borders)

	# LABEL ARRAY
	dw_arr = dw_reader.read(1)

	# FIND CATEGORIES/ADJUST LABEL
	dw_water_mask  = dw_arr == 1 #water
	dw_nodata_mask = dw_arr == 0 #no data
	dw_land_mask   = ~(dw_water_mask | dw_nodata_mask) #land/everything else
	dw_arr[dw_water_mask]  = 255
	dw_arr[dw_land_mask]   = 0
	dw_arr[dw_nodata_mask] = 128
	dw_arr_3d = np.repeat(dw_arr[np.newaxis,:,:],repeats=3,axis=0)

	# COPY ADJUSTED LABEL TO (LARGER) S2-SIZED ARRAY
	out_arr = np.zeros((3,s2_reader.height,s2_reader.width)).astype(np.uint8)
	out_arr[:,s2_borders['top']:s2_borders['bottom'],s2_borders['left']:s2_borders['right']] = dw_arr_3d	

	# LINES TO DRAW
	red_line       = np.zeros((3,chip_size))
	red_line[0]    = 255
	yellow_line    = np.zeros((3,chip_size))
	yellow_line[0] = 255
	yellow_line[1] = 255

	# PARAM FOR SELECTION OF GOOD/BAD CHIPS
	WATER_MIN = chip_size*chip_size//4 
	WATER_MAX = chip_size*chip_size - WATER_MIN

	# SET OUT PATH
	dw_tile_id = dw_img_path.split('/')[-1].split('.')[0]
	out_path   = f"./figures/{dw_tile_id}_GRID.tif"


	for k,(rowcol,w) in enumerate(s2_windows):

		# LOAD LABEL WINDOW
		# dw_win_array = dw_reader.read(1,window=dw_windows[k][1])
		w2 = dw_windows[k][1]
		dw_win_array = dw_arr[w2.row_off:w2.row_off+chip_size, w2.col_off:w2.col_off+chip_size]

		# Copy label
		# out_arr[0,w.row_off:w.row_off+chip_size,w.col_off:w.col_off+chip_size] = dw_win_array
		# out_arr[1,w.row_off:w.row_off+chip_size,w.col_off:w.col_off+chip_size] = dw_win_array
		# out_arr[2,w.row_off:w.row_off+chip_size,w.col_off:w.col_off+chip_size] = dw_win_array

		# WINDOW IN NO-DATA -- plot just label
		if (dw_win_array==128).any():
			continue

		# WINDOW BAD RATIO -- plot label & lines
		n_water = (dw_win_arr==255).sum()
		if n_water < WATER_MIN or n_water > WATER_MAX:
			# Set lines
			for i in range(3): #3px width
				out_arr[:, w.row_off+i, w.col_off:w.col_off+chip_size]             = red_line # top row
				out_arr[:, w.row_off+chip_size-(i), w.col_off:w.col_off+chip_size] = red_line # bottom
				out_arr[:, w.row_off:w.row_off+chip_size, w.col_off+i]             = red_line # left
				out_arr[:, w.row_off:w.row_off+chip_size, w.col_off+chip_size-i]   = red_line # right
			continue

		# WINDOW GOOD RATIO -- plot label & lines
		# Set lines
		for i in range(3): #3px width
			out_arr[:, w.row_off+i, w.col_off:w.col_off+chip_size]             = yellow_line # top row
			out_arr[:, w.row_off+chip_size-(i), w.col_off:w.col_off+chip_size] = yellow_line # bottom
			out_arr[:, w.row_off:w.row_off+chip_size, w.col_off+i]             = yellow_line # left
			out_arr[:, w.row_off:w.row_off+chip_size, w.col_off+chip_size-i]   = yellow_line # right
	

	out_ptr.write(out_arr)
	out_ptr.close()
	s2_reader.close()
	dw_reader.close()
	print(f"file written to {out_path}")


################################################################################
# CHIPS
################################################################################
def plot_chip_bands(chip_path):

	assert os.path.isfile(chip_path), f"No file in {chip_path}"

	# LOAD & SEPARATE BANDS
	r,g,b,nir = Image.open(chip_path).split()
	rgb = Image.merge(mode='RGB',bands=[r,g,b])
	false_ir = Image.merge(mode='RGB',bands=[nir,r,g])

	#PATHS
	rgb_path      = f"./figures/{chip_path.split('/')[-1].replace('.tif','_rgb.tif')}"
	false_ir_path = f"./figures/{chip_path.split('/')[-1].replace('.tif','_nir.tif')}"

	# SAVE
	rgb.save(rgb_path)
	false_ir.save(false_ir_path)

	# LOG/STDOUT
	print(f"File saved to {rgb_path}.")
	print(f"File saved to {false_ir_path}.")


def plot_chip_label(chip_path):
	'''
	Copy label in dataset to same directory.
	'''

	lbl_path = chip_path.replace("_B0X.tif","_LBL.tif")
	img      = Image.open(lbl_path)
	out_path = f"./figures/{lbl_path.split('/')[-1]}"
	img.save(out_path)
	print(f"File saved to {out_path}.")


def plot_chip_panel(chip_path):
	'''
	Single figure. Plot RGB, false-color, and label side by side.
	'''

	# CHECK PATH
	assert os.path.isfile(chip_path), f"No file in {chip_path}"

	# LOAD & SEPARATE BANDS
	r,g,b,nir = Image.open(chip_path).split()
	rgb = Image.merge(mode='RGB',bands=[r,g,b])
	false_ir = Image.merge(mode='RGB',bands=[nir,r,g])
	label = Image.open(chip_path.replace("_B0X.tif","_LBL.tif"))

	# SET OUT PATH
	chip_id  = chip_path.split('/')[-1].split("_B0X.tif")[0]
	out_path = f"./figures/{chip_id}_panel.png"

	# PLOT
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(rgb)
	# axes[0].set_title("RGB Bands")
	axes[0].axis("off")
	axes[1].imshow(false_ir)
	# axes[1].set_title("False-color IR")
	axes[1].axis("off")
	axes[2].imshow(label,cmap='gray')
	# axes[2].set_title("Label")
	axes[2].axis("off")
	plt.tight_layout()
	plt.savefig(out_path, bbox_inches="tight", dpi=300)
	plt.close()

	# if need to adjust brightness
	# r = np.array(red)
	# g = np.array(grn)
	# b = np.array(blu)
	# n = np.array(nir)
	# factor = 255
	# new_r = Image.fromarray(((r - r.min())/(r.max()-r.min()) * factor).astype(np.uint8))
	# new_g = Image.fromarray(((g - g.min())/(g.max()-g.min()) * factor).astype(np.uint8))
	# new_b = Image.fromarray(((b - b.min())/(b.max()-b.min()) * factor).astype(np.uint8))
	# new_n = Image.fromarray(((n - n.min())/(n.max()-n.min()) * factor).astype(np.uint8))
	# rgb = Image.merge('RGB',[new_r,new_g,new_b])
	# rgb.save("./chip_rgb.jpg")
	# nrg = Image.merge('RGB',[new_n,new_r,new_g])
	# nrg.save("./chip_ngr.jpg")

	# LOG/STDOUT
	print(f"File saved to {out_path}.")


################################################################################
# BARS & HISTOGRAMS -- CHIPS
################################################################################
def plot_chip_band_histogram(chip_path):

	# CHECK DIR
	assert os.path.isfile(chip_path), f"No chip file found in path {chip_path}"
	red,grn,blu,nir = Image.open(chip_path).split()

	chip_id  = chip_path.split('/')[-1].split("_B0X.tif")[0]
	out_path = f"./figures/{chip_id}_hist.png"

	# PLOT
	bins = 256
	y_max = 25000 # < 65536?

	fig,axes = plt.subplots(nrows=1,ncols=4,figsize=fig_size_wide,dpi=300,sharey=True)
	axes[0].hist(np.array(nir).flatten(),bins=bins,histtype='bar',color='darkred')
	axes[0].set_title("Near-infrared")
	axes[0].set_xlim(0,255)
	axes[0].set_ylabel("Count")
	axes[1].hist(np.array(red).flatten(),bins=bins,histtype='bar',color='red')
	axes[1].set_title("Red")
	axes[1].set_xlim(0,255)
	axes[2].hist(np.array(grn).flatten(),bins=bins,histtype='bar',color='green')
	axes[2].set_title("Green")
	axes[2].set_xlim(0,255)
	axes[3].hist(np.array(blu).flatten(),bins=bins,histtype='bar',color='blue')
	axes[3].set_title("Blue")
	axes[3].set_xlim(0,255)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

	print(f"File saved to {out_path}.")


################################################################################
# BARS & HISTOGRAMS -- DATASET DISTRIBUTION
################################################################################
def plot_chips_per_raster(chip_dir):
	'''
	Histogram. Distribution of chips per raster.
	'''	
	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No chip file found in path {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te
	chip_rasters = [_[0:-19] for _ in chips]
	
	# FIND UNIQUE RASTER NAMES AND COUNT
	rasters,chips_per_raster = np.unique(chip_rasters,return_counts=True)

	# PLOT
	plt.figure(figsize=fig_size_normal)
	plt.hist(chips_per_raster,histtype='bar',bins=100)
	plt.title(f"Distribution of Chips Produced by Each Raster (N={len(rasters)})")
	# plt.ylabel("Count")
	plt.xlabel("Chips") #this is correct
	plt.tight_layout()
	plt.savefig('./figures/chips_per_raster.png',dpi=300)
	plt.close()


def plot_chips_per_tile(chip_dir):
	'''
	Bar plot. Count of chips in each of the 21 tiles.
	'''

	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No chip file found in path {chip_dir}"

	# LIST CHIPS IN DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te

	# GET TILES IN LIST
	chip_tiles   = [_[32:-19] for _ in chips]	


	# FIND UNIQUE TILES AND COUNT
	tiles, chips_per_tile = np.unique(chip_tiles,return_counts=True) 

	# PLOT
	plt.figure(figsize=fig_size_wide)
	plt.bar(tiles,chips_per_tile,color='C0')
	plt.ylabel("Chips")
	plt.xlabel("Tile")
	plt.xticks(tiles,rotation=90)	
	plt.tight_layout()
	plt.savefig("./figures/chips_per_tile.png",dpi=300)
	plt.close()


def plot_chips_per_month(chip_dir):
	'''
	Bar plot.
	'''

	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No chip file found in path {chip_dir}"

	# CHIPS IN DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te

	# GET DATES
	# chip_dates  = [_[0:8] for _ in chips]
	chip_months = [_[0:6] for _ in chips]

	# GET UNIQUE DATES AND COUNT
	# dates, chips_per_date = np.unique(chip_tiles,return_counts=True)
	months, chips_per_month = np.unique(chip_months,return_counts=True)

	# PLOT
	month_ticks = [datetime.strptime(_,"%Y%m").strftime("%b-'%y") for _ in months]
	plt.figure(figsize=fig_size_normal)
	plt.bar(month_ticks,chips_per_month,color='C0')
	plt.ylabel("Chips")
	plt.xlabel("Month")
	plt.xticks(rotation=90)		
	plt.tight_layout()
	plt.savefig("./figures/chips_per_month.png",dpi=300)
	plt.close()


def plot_chips_per_week(chip_dir):
	'''
	Bar plot. Count of chips per week, for 52 weeks of year 2025.
	'''

	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No chip file found in path {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te	

	# GET DATES, EXTRACT WEEK
	chip_dates     = [_[0:8] for _ in chips]
	chip_dates_obj = [datetime.strptime(_,"%Y%m%d") for _ in chip_dates]	
	chip_weeks     = [f"{_.isocalendar().year}-{_.isocalendar().week:02}" for _ in chip_dates_obj]

	# GET UNIQUE WEEKS AND COUNT CHIPS
	weeks,chips_per_week = np.unique(chip_weeks,return_counts=True)

	# PLOT
	plt.figure(figsize=fig_size_wide)
	plt.bar(weeks,chips_per_week,color='C0')
	plt.ylabel("Chips")
	plt.xlabel("Week")
	plt.xticks(rotation=90)		
	plt.tight_layout()
	plt.savefig("./figures/chips_per_week.png",dpi=300)
	plt.close()


def plot_rasters_per_week(chip_dir):
	'''
	Bar plot.
	'''
	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No data directory in {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te
	chip_rasters = [_[0:-19] for _ in chips]
	
	# FIND UNIQUE RASTER NAMES
	rasters,chips_per_raster = np.unique(chip_rasters,return_counts=True)

	# GET DATES, EXTRACT WEEK
	raster_dates     = [_[0:8] for _ in rasters]
	raster_dates_obj = [datetime.strptime(_,"%Y%m%d") for _ in raster_dates]
	raster_weeks     = [f"{_.isocalendar().year}-{_.isocalendar().week:02}" for _ in raster_dates_obj]

	# GET UNIQUE WEEKS
	weeks,rasters_per_week = np.unique(raster_weeks,return_counts=True)	

	# PLOT
	plt.figure(figsize=fig_size_wide)
	plt.bar(weeks,rasters_per_week,color='C0')
	plt.ylabel("Rasters")
	plt.xlabel("Week")
	plt.xticks(rotation=90)		
	plt.tight_layout()
	plt.savefig("./figures/rasters_per_week.png",dpi=300)
	plt.close()


def plot_rasters_per_tile(chip_dir):
	'''
	Bar plot.
	'''
	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No data directory in {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te

	# FIND UNIQUE RASTER NAMES
	rasters = [_[0:-19] for _ in chips]
	unique_rasters,chips_per_raster = np.unique(rasters,return_counts=True)

	# GET TILES
	tiles = [_[-6:] for _ in unique_rasters]		

	# GET UNIQUE TILES AND COUNT
	unique_tiles, rasters_per_tile = np.unique(tiles,return_counts=True) 

	# PLOT
	plt.figure(figsize=fig_size_wide)
	plt.bar(unique_tiles,rasters_per_tile,color='C0')
	plt.ylabel("Raster")
	plt.xlabel("Tiles")
	plt.xticks(rotation=90)		
	plt.tight_layout()
	plt.savefig("./figures/rasters_per_tile.png",dpi=300)
	plt.close()


def plot_chips_per_window(chip_dir):
	'''
	Histogram and average. Count of chips produced by each window.
	'''
	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No data directory in {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te


	windows = ['_'.join([_[32:38],_[-13:-8]]) for _ in chips]
	unique_windows,chips_per_window = np.unique(windows,return_counts=True)

	print(f"Mean chips per window: {chips_per_window.mean()}")

	# PLOT
	plt.figure(figsize=fig_size_normal)
	plt.hist(chips_per_window,histtype='bar',bins=50)
	plt.title(f"Distribution of Chips Produced by Each Window (N={len(unique_windows)})")
	# plt.ylabel("Count")
	plt.xlabel("Chips")
	plt.tight_layout()
	plt.savefig('./figures/chips_per_window.png',dpi=300)
	plt.close()


def plot_window_intervals(chip_dir):
	'''
	Histogram. Average revisit (distance between dates in days) of each window.
	'''
	# CHECK DIR
	assert os.path.isdir(chip_dir), f"No data directory in {chip_dir}"

	# LIST DIR
	chips_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chips_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chips_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")
	chips    = chips_tr + chips_va + chips_te

	# ARRAYS
	windows = np.array(['_'.join([_[32:38],_[-13:-8]]) for _ in chips])
	dates   = np.array([_[0:8] for _ in chips])
	unique_windows = np.unique(windows)

	# CALCULATE REVISIT TIMES PER WINDOW
	all_revisit_times = []
	for w in unique_windows:
		window_dates = dates[windows == w]
		dates_obj    = sorted([datetime.strptime(_,"%Y%m%d") for _ in window_dates])
		# day_of_year  = [int(_.strftime("%j")) for _ in dates_obj]
		revisit_time = [(dates_obj[i] - dates_obj[i-1]).days for i in range(1,len(dates_obj))]
		all_revisit_times += revisit_time


	mean_revisit = np.array(all_revisit_times).mean()
	print(f"Mean window revisit time: {mean_revisit}")

	# PLOT
	plt.figure(figsize=fig_size_normal)
	plt.hist(all_revisit_times,histtype='bar',bins=100)
	plt.title(f"Revisit Times per Individual Chip Location")
	# plt.ylabel("Count")
	plt.xlabel("Revisit time (in days)")
	plt.tight_layout()
	plt.savefig('./figures/window_revisit.png',dpi=300)
	plt.close()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',required=True,default=None,help='Dataset (chip) directory.')
	args = parser.parse_args()

	# if args.chip_dir is not None:
	assert os.path.isdir(args.chip_dir), f"Chip dir {args.chip_dir} not found."
		# if args.chip_dir[-1] == '/':
			# args.chip_dir = args.chip_dir[:-1]
	args.chip_dir = args.chip_dir.rstrip('/')

	return args

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	args = parse_args()

	# sample_dw_tile = "20250627T184941_20250627T185915_T10SEJ.tif"
	# dw_tile_path = f"../{sample_dw_tile}"
	# plot_tile_label(dw_tile_path)
	# plot_tile_label_original(dw_tile_path)

	# POLYGONS/GEOGRAPHICAL DISTRIBUTION OF TILES
	if os.path.isfile(STATE_SHP):
		plot_tile_polygons()
		plot_tile_polygons_with_split()

	# ONE CHIP
	chips_va = glob("*_B0X.tif",root_dir=f"{args.chip_dir}/validation")
	sample_chip = f"validation/{chips_va[0]}"
	chip_path = f"{args.chip_dir}/{sample_chip}"
	plot_chip_panel(chip_path)
	plot_chip_bands(chip_path)
	plot_chip_label(chip_path)
	plot_chip_band_histogram(chip_path)

	# DATASET PLOTS
	plot_chips_per_raster(args.chip_dir)
	plot_chips_per_tile(args.chip_dir)
	plot_chips_per_month(args.chip_dir)	
	plot_chips_per_week(args.chip_dir)
	plot_rasters_per_week(args.chip_dir)
	plot_rasters_per_tile(args.chip_dir)
	plot_chips_per_window(args.chip_dir)
	plot_window_intervals(args.chip_dir)
