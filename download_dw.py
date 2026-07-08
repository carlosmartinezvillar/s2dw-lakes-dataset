'''
Download DynamicWorld V1 products from GEE to a Google Drive folder.
'''
import os
import time
import numpy as np
import ee #earth engine Python API
import rasterio as rio
import xml.etree.ElementTree as ET
import sys
import subprocess sp

S2_DIR = None

def get_gee_id(safe_path):
	'''
	Take Sentinel-2 .SAFE string. Return respective DynamicWorld id.
	'''
	subdir = glob.glob("*",root_dir=f"{safe_path}/GRANULE")[0]
	datastrip = subdir.split('_')[-1]
	date,tile = safe_path.split('/')[-1].split('_')[2:6:3]
	gee_id = f"{date}_{datastrip}_{tile}"
	return gee_id


def get_band_file_path(safe_path):
	band_regex = f"GRANULE/*/IMG_DATA/R10m/*_B02_10m.jp2"
	path = f"{safe_path}/{glob.glob(band_regex,root_dir=safe_path)}"	
	return path


def select_shift_unmask(gee_img):
	lbl_shifted_unmasked = gee_img.select('label').add(1).unmask().uint8()
	return lbl_shifted_unmasked


def create_task(ee_image: ee.Image, ee_id: str, s2_rdr: rio.io.DatasetReader) -> ee.batch.Task:
	'''
	Create a batch export task in GEE.

	Parameters
	----------
	ee_image : ee.Image
		A google earth engine object pointing to a DynamicWorld file
	ee_id : str
		A string with the GEE id of ee_image (following GEE convention)
	s2_rdr: rio.io.DatasetReader
		A rasterio object pointing to a local sentinel 2 image

	Returns
	-------
	task : ee.batch.Task
		Task object that can be used to start and check export task to google drive
	'''

	if type(s2_rdr) is rio.io.DatasetReader:
		s2_crs   = s2_rdr.crs.to_string()
		s2_crs_t = str([int(_) for _ in s2_rdr.transform[0:6]]).strip('[]')
	
	if s2_rdr is None:
		s2_rdr   = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + ee_id).select('B2')
		s2_crs   = s2_rdr.getInfo()['bands'][0]['crs'] #from GEE
		s2_crs_t = str(s2_rdr.getInfo()['bands'][0]['crs_transform']).strip('[]') #from GEE

	task = ee.batch.Export.image.toDrive(
		image=ee_image,
		description=ee_id,
		folder="DW_LABELS",
		fileNamePrefix=ee_id,
		scale=10,
		crs=s2_crs,
		crsTransform=s2_crs_t,
		maxPixels=1e9,
		fileFormat='GeoTIFF'
		)

	return task


def start_task(task: ee.batch.Task ,ee_id: str):
	'''
	Launch a GEE task
	'''
	print(f"Launching Drive task {ee_id}...")
	try:
		task.start()
	except Exception as e:
		print("Error launching task. Skipping...")


def parse_args():
	# CONFIG
	parser = argparse.ArgumentParser(
		prog="download_dw.py",
		description="Download DynamicWorld V1 to a Google Drive folder.")
	parser.add_argument('--s2-dir',default=None,
		help="Source directory for raw Sentinel-2 products.")

	# READ
	args = parser.parse_args()
	if args.s2_dir is None:
		print("No S2 dir given.")
		sys.exit(1)
	if not os.path.isdir(args.s2_dir):
		print("S2 path given does not exist.")
		sys.exit(1)

	# SET
	global S2_DIR
	S2_DIR = args.s2_dir


if __name__ == '__main__':
	########## ARGS ##########
	parse_args()

	########## GEE & CREDENTIALS ##########
	print("Running ee.Initialize()...")
	ee.Initialize()

	########## CREATE TASKS ##########
	# CHECK PREVIOUS PENDING TASKS
	prev_tasks = [i['description'] for i in ee.batch.data.getTaskList()]

	# S2 DIRS
	safe_regex   = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE"
	safe_folders = glob.glob(safe_regex,root_dir=S2_DIR) #returns ['/eodata/.../*.SAFE']

	ee_ids = []
	tasks  = []

	for i,safe_path in enumerate(safe_folders)

		ee_id = get_gee_id(safe_path)

		if ee_id in prev_tasks:
			continue

		# CREATE GEE REF
		ee_ids.append(ee_id)
		ee_img = ee.Image('GOOGLE/DYNAMICWORLD/V1/' + ee_id) #check for return value of wrong ref
		ee_img = select_shift_unmask(ee_img)

		# GET S2 CRS
		s2_band_path = get_band_file_path(safe_path)
		s2_reader    = rio.open(s2_band_path,'r')

		#CREATE TASK
		print(f"Creating Drive task for product {ee_id}")		
		task = create_task(ee_img,ee_id,s2_reader)
		tasks.append(task)

		# CLEAN UP
		s2_reader.close()

	########## LAUNCH TASKS ##########
	for task,name in zip(tasks,ee_ids):
		start_task(task,name)
