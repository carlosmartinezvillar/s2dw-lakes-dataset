'''
Download DynamicWorld V1 products from GEE to a Google Drive folder.
This script uses individual user credentials set by ee, the Google Earth Engine API.
'''
import os
import time
import ee #earth engine Python API
import rasterio as rio
import sys
import argparse
import glob
import json
from google.oauth2.credentials import Credentials
import ee

S2_DIR = None
DRIVE_FOLDER = "DW_LABELS"

def get_gee_id(safe_path:str) -> str:
	'''
	Get DynamicWorld id from Sentinel-2 .SAFE string and subfolder.
	'''
	subdir = glob.glob("*",root_dir=f"{safe_path}/GRANULE")[0]
	datastrip = subdir.split('_')[-1]
	date,tile = safe_path.split('/')[-1].split('_')[2:6:3]
	gee_id = f"{date}_{datastrip}_{tile}"
	return gee_id


def get_band_file_path(safe_path:str) -> str:
	'''
	Get absolute path for band file from /**/*.SAFE folder.
	'''
	band_regex = f"GRANULE/*/IMG_DATA/R10m/*_B02_10m.jp2"
	path = f"{safe_path}/{glob.glob(band_regex,root_dir=safe_path)[0]}"	
	return path


def select_shift_unmask(ee_image:ee.Image) -> ee.Image:
	'''
	Get label, replace masking with simple +1 shift, and set to 8 bits.
	'''
	shifted_unmasked = ee_image.select('label').add(1).unmask().uint8()
	return shifted_unmasked


def create_task(ee_image:ee.Image, ee_id:str, s2_rdr:rio.io.DatasetReader) -> ee.batch.Task:
	'''
	Create a batch export task in GEE.

	Parameters
	----------
	ee_image : ee.Image
		A google earth engine object pointing to a DynamicWorld file
	ee_id : str
		A string with the GEE id of ee_image (following GEE convention)
	s2_rdr: rio.io.DatasetReader
		Rasterio object pointing to a local Sentinel-2 band file

	Returns
	-------
	task : ee.batch.Task
		Task object that can be used to start and check export task to google drive
	'''

	# Get CRS from a Sentinel-2 band image
	if type(s2_rdr) is rio.io.DatasetReader:
		s2_crs   = s2_rdr.crs.to_string()
		# s2_crs_t = str([int(_) for _ in s2_rdr.transform[0:6]]).strip('[]')
		s2_crs_t = list(s2_rdr.transform[0:6])

	# Alternatively, get CRS from S2 image in EE
	if s2_rdr is None:
		s2_rdr   = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + ee_id).select('B2')
		info     = s2_rdr.getInfo()
		s2_crs   = info['bands'][0]['crs']
		s2_crs_t = str(info['bands'][0]['crs_transform']).strip('[]')

	# SET TASK
	task = ee.batch.Export.image.toDrive(
		image=ee_image,
		description=ee_id,
		folder=DRIVE_FOLDER,
		fileNamePrefix=ee_id,
		scale=10,
		crs=s2_crs,
		crsTransform=s2_crs_t,
		maxPixels=1e9,
		fileFormat='GeoTIFF'
		)

	return task


def start_task(task:ee.batch.Task ,ee_id:str):
	'''
	Launch a GEE task
	'''
	try:
		task.start()
		print(f"Launched Drive task for {ee_id}...")		
	except Exception as e:
		print(f"Error launching task for {ee_id}.\nSkipping...")


def parse_args():
	# CONFIG
	parser = argparse.ArgumentParser(
		prog="download_dw.py",
		description="Download DynamicWorld V1 products to a Google Drive folder.")
	parser.add_argument('--s2-dir',default=None,
		help="Source directory of Sentinel-2 products.")

	# READ
	args = parser.parse_args()
	assert args.s2_dir is not None, "No S2 dir given."
	assert os.path.isdir(args.s2_dir), "S2 path given does not exist."

	# SET
	global S2_DIR
	S2_DIR = args.s2_dir


if __name__ == '__main__':
	########## ARGS ##########
	parse_args()

	########## I.GEE & CREDENTIALS ##########
	print("Running ee.Initialize()...")
	CREDENTIALS_PATH = '/root/.config/earthengine/credentials'
	with open(CREDENTIALS_PATH, 'r') as f:
		cred_data = json.load(f)

	# Format the token payload into Google OAuth2 credentials object
	# Legacy files use 'refresh_token', 'client_id', and 'client_secret'
	scoped_credentials = Credentials(
		token=None,
		refresh_token=cred_data.get('refresh_token'),
		client_id=cred_data.get('client_id'),
		client_secret=cred_data.get('client_secret'),
		token_uri=ee.oauth.TOKEN_URI, # Maps to 'https://googleapis.com'
		scopes=ee.oauth.SCOPES
	)

	# Initialize forcing these specific credentials and your project
	ee.Initialize(
		credentials=scoped_credentials,
		project='s2dw-lakes-masks'
	)
	# ee.Initialize()

	########## II.CREATE TASKS ##########
	# CHECK PREVIOUS PENDING TASKS
	prev_tasks = [i['description'] for i in ee.batch.data.getTaskList()]

	# S2 DIRS
	safe_regex   = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE"
	safe_folders = glob.glob(safe_regex,root_dir=S2_DIR) #returns ['/eodata/.../*.SAFE']

	ee_ids = []
	tasks  = []
	missing = []

	for safe_path in safe_folders:

		ee_id = get_gee_id(safe_path)

		if ee_id in prev_tasks:
			continue

		try:
			ee.data.getAsset(f"GOOGLE/DYNAMICWORLD/V1/{ee_id}") # Check metadata w/o loading
		except ee.EEException:
			print(f"No asset: {ee_id}")
			missing.append(ee_id)
			continue		

		# CREATE GEE REF
		ee_ids.append(ee_id)
		ee_img = ee.Image(f"GOOGLE/DYNAMICWORLD/V1/{ee_id}")
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

	########## III.LAUNCH TASKS ##########
	for task,name in zip(tasks,ee_ids):
		start_task(task,name)
