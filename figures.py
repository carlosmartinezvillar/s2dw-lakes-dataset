'''
Several plots and figures.
'''
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

STATE_SHP = "../"

def plot_tile_geometries(us_geom,out_path):
	'''
	us_geom: a .shp file for US state geometries.
	'''
	# TILES IN DATASET
	tile_geometries_file = "./search/search_results_geometries.tsv"
	dropped_tiles = ["T11SKD","T11TKE"]

	#LOAD AS GEODATAFRAME

	#SINGLE ENTRIES PER TILE

	#PLOT
	pass


def plot_tile_bands(out_path):
	'''
	Plot an RGB image.
	'''
	pass


def plot_tile_label(out_path):
	'''
	Plot a label.
	'''
	pass


def plot_chip_bands(out_path):
	pass


def plot_chip_label(out_path):
	pass


def plot_chip(out_path):
	'''
	Single figure. RGB, false-color, and label.
	'''
	pass


def plot_tile_label_windows(out_path):
	'''
	Water mask with windows (chip positions) highlighted.
	'''
	pass


def rasters_per_week(out_path):
	'''
	Bar plot.
	'''
	pass


def rasters_per_tile(out_path):
	'''
	Bar plot.
	'''
	pass


def chips_per_raster(out_path):
	'''
	Histogram. Distribution of # chips per single raster.
	'''
	pass


if __name__ == '__main__':
	pass