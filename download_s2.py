import json
import yaml
import os
import xml.etree.ElementTree as ET
import requests
import multiprocessing
import subprocess as sp
import time

class Downloader():
	"""
	Explanation of the class...
	"""
	def __init__(self,out_dir):

		#CHECK ARGS
		if not os.path.isdir(out_dir):
			print("Output dir not found.")
			raise FileNotFoundError
		if out_dir[-1] == '/':
			out_dir = out_dir[0:-1]
		self.out_dir = out_dir

		#SATELLITE PARAMETERS
		self.instrument  = "MSI"
		self.productType = "S2MSI2A"
		self.sensorMode  = None #some "null" some INS-NOBS for S2 ¿?

		#AOI PARAMETERS
		self.cloudCover = "5.00"
		self.startDate  = "2025-01-01T00:00:00.000Z"
		self.endDate    = "2026-01-01T00:00:00.000Z"
		self.geometry   = "POINT(00.00,00.00)"
		self.bands      = ["B02","B03","B04","B08"]

		#JSON RETURN PARAMETERS
		self.maxRecords = 100
		self.sortParam  = "startDate"
		self.sortOrder  = "ascending"

		#QUERY & SEARCH RESULTS
		self.query    = None
		self.names    = [] #["*.SAFE"]
		self.s3_ids   = [] #["eodata/Sentinel-2/MSI/.../*.SAFE"]
		self.polygons = [] #[{"type":"Polygon","coordinates":[[[]]]}]
		self.statuses = []
		self.baselines = []

		#DOWNLOAD
		self.RCLONE_MAX = "4"
		self.ODATA_BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"

		# S3 QUEUE
		self.s3_ids_filtered = self.s3_id
		self.s3_band_paths = []


	def build_odata_query(self):
		'''
		Set up an OData query for CDSE using object parameters.
		'''
		#Filters
		collection  = "Collection/Name eq 'SENTINEL-2'"
		producttype = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'" \
						f" and att/OData.CSC.StringAttribute/Value eq '{self.productType}')"
		aoi         = f"OData.CSC.Intersects(area=geography'SRID=4326;{self.geometry}')"
		clouds      = f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'" \
						f" and att/OData.CSC.DoubleAttribute/Value le {self.cloudCover})"
		date_start  = f"ContentDate/Start gt {self.startDate}"
		date_final  = f"ContentDate/Start lt {self.endDate}"

		#Join filters
		filters_str = " and ".join([collection,producttype,aoi,clouds,date_start,date_final])

		#Join query params
		req_filter = f"$filter={filters_str}"
		req_top    = f"$top={self.maxRecords}"
		req_skip   = f"$skip=0"
		req_count  = "$count=True"
		req_order  = "$orderby=ContentDate/Start desc"
		req = "&".join([req_filter,req_top,req_skip,req_order,req_count])

		self.query = self.ODATA_BASE_URL + req
		return self.query


	def search_odata(self,query=None):
		'''
		Search pages for a single geometry
		'''
		#Get first page
		req  = self.build_odata_query()
		resp = requests.get(req)

		#Check response
		if resp.status_code !=200:
			print(resp.text)
			return

		#convert to json
		resp_json = resp.json()
		print(f"Products found: {resp_json['@odata.count']}.")

		#No matching results
		if resp_json['@odata.count'] == 0:
			return

		#handle page count
		n_pages = resp_json['@odata.count'] // self.maxRecords
		if resp_json['@odata.count'] % self.maxRecords > 0:
			n_pages += 1
		current_page = 1

		#iterate through pages
		while True:

			#stdout
			print(f"Retrieving pages [{current_page}/{n_pages}]")

			#store entries
			for entry in resp_json['value']:
				self.names.append(entry['Name'])
				self.polygons.append(entry['Footprint'])
				self.s3_ids.append(entry['S3Path'].lstrip('/')) #remove leading '/'

			# more pages?
			if "@odata.nextLink" not in resp_json:
				break

			# Get next page
			next_page = resp_json['@odata.nextLink']
			resp      = requests.get(next_page)

			if resp.status_code !=200:
				print("STATUS CODE!=200.")
				print(resp.text)
				break

			resp_json = resp.json()			
			current_page += 1


	def drop_tiles_from_queue(self,tile_str):
		safe_tiles = [s.split('/')[-1].split('_')[-2] for s in self.s3_ids_filtered]
		mask = np.array(safe_tiles) == tile_str
		self.s3_ids_filtered = self.s3_ids_filtered[~mask]
		

	def log_search(self):
		#LOG THE SEARCH -- links
		log = [f"{product}\t{s3}" for product,s3 in zip(self.names,self.s3_ids)]
		with open(f"{self.out_dir}/search_results.tsv",'w') as fp:
			fp.write("\n".join(log))

		#LOG THE SEARCH -- geo
		log = [f"{product}\t{poly}" for product,poly in zip(self.names,self.polygons)]
		with open(f"{self.out_dir}/search_results_geometries.tsv",'w') as fp:
			fp.write("\n".join(log))

		#LOG DOWNLOAD QUEUE -- include files for S3 client (rclone)
		for s3folder in self.s3_ids_filtered:
			for band in self.bands:
				self.s3_band_paths.append(f"{s3folder}/GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2")

		with open(f"{self.out_dir}/download_queue.txt",'w') as fp:
			fp.write(str("\n".join(self.s3_band_paths)))


	def download_rclone_queue(self,queue_file):
		'''
		Download products via S3.
		'''
		#download -- this does only a dry run. Remove dry-run flag to actually download.
		proc0 = sp.run([
			"rclone","copy",
			"--include-from",queue_file,
			"esa:",self.out_dir,"-P",
			"--transfers",self.RCLONE_MAX,"--dry-run"])


if __name__ == '__main__':

	D = Downloader("./search")

	with open('./search/centroids.txt','r') as fp:
		centroids = fp.readlines()
	centroids = [tuple(s.rstrip().split(',')) for s in centroids]

	for lat,lon in centroids:
		D.geometry = f"POINT({lon} {lat})"
		print(f"Searching {D.geometry} ...")
		D.search_odata()

	D.drop_tiles_from_queue("T11SKD")
	D.drop_tiles_from_queue("T11TKE")
	D.log_search()

