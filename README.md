# s2dw-lakes-dataset
Pre-processing scripts and evaluation models for the S2DW dataset for inland water body segmentation.

Paper link: (coming soon)

## About

## Overview

The overall processing pipeline for this data looks somwhat like this:

```
(centroids.txt)
	|
	v
[Search Sentinel-2] -> [Drop Bad Tiles]
							|
							v
					(download_queue.txt)
```
```
(download_queue.txt)
	|
	v
[S3:Download Sentinel-2] -> [GEE:Search & Download DW]
									|
									v
							{Raw S2+DW data (~440 GB)}
```
```
{Raw S2+DW data}
	|
	v
[Align & correct] -> [Chip & Filter] -> [Train/val/test split]
											|
											v
										*Final Dataset*
```

## Contents/Scripts
This repository is composed of several scripts for both the dataset and the benchmark models ran. The relevant scripts for the dataset are:

- **`download_s2.py`**: Ingests the points in `search/centroids.txt` to search the Sentinel-2 L2A archive via OData API, and logs this search into `.tsv` files within `search/`, and `search/download_queue.txt`. Drops two MGRS tiles from the list of results before writing these files. The outputted `.txt` queue has absolute paths to 10-meter band files in the ESA archive (to be downloaded via S3).

- `download_dw.py`: Takes the path to a local directory containing Sentinel-2 files to create corresponding DynamicWorld V1 id's. This requires the Google Earth Engine API to search for these ids, and to launch tasks copying matching results to a Google Drive/Google Cloud folder. 

- `chip.py`: Process each S2-Dw pair to produce chips. Processing is done in batches, copying a limited number (`chunk_size=50`) raster pairs to lower limit disk use in containerized/slow-disk environments.

- `split.py`: Splits the resulting dataset from `chips.py` (stored as a single directory) into training, testing, and validation sets using MGRS tile locations. Resulting split is stored as a new copy. 

- `stats.py`: Overall mean and standard deviation of the bands in the training set. Calculates distribution of water vs. non-water pixels (~47/53%).

Some additional files are not crucial to reproduce the dataset but generate statistics used for model training (standardized band inputs & class weights), and summarize the dataset:

- `split.txt`: Resulting split (i.e., tiles allocated) from `split.py`.

- `figures.py`: Plots several figures summarizing dataset distribution.

Remaining files are:

- `models/`: Subdirectory containing scripts for some basic benchmark CNN/ViT training.
- `search/`: Inputs and outputs for `download_s2.py`.
	- `centroids.txt`: Input latitude and longitude to search Sentinel-2 products.
	- `download_queue.txt`: The actual downloaded Sentinel-2 images.
	- `drop.txt`: Tiles dropped.
	- `search_results.tsv`: Search result .SAFE folders and S3 paths.
	- `search_results_geometries.tsv`: .SAFE paths and polygon/geometries for each raster.

- `figures/`: Dataset figures


## How-to
### Dataset
To reproduce the dataset, run in this order:

```
[download_s2.py] -> [download_dw.py] -> [chip.py] -> [split.py]
```
To search and download Sentinel-2 products run

```
python3 download_s2.py
```

To download masks, replace `S2_DIR` with the directory containing the previously downloaded Sentinel-2 products.
 
```
python3 download_dw.py --s2-dir {S2_DIR}
```
Once downloaded through GEE, copy these files to a local directory (approx. ~4.5GB).

Chip the 10980x10980 rasters into 256x256 chips using:

```
python3 chip.py --work-dir {WORK_DIR} --chip-dir {CHIP_DIR} --s2-dir {S2_DIR} --dw-dir {DW_DIR}
```

where `CHIP_DIR` is the output directory to store the newly created chips. `S2_DIR` and `DW_DIR` should point to the paths of the downloaded Sentinel-2 and DynamicWorld V1 files, respectively. `WORK_DIR` must be set to any temporary working directory with fast access and sufficient memory.

### Benchmark models
To fit any model to the training dataset run

```
python3 train.py --data-dir {DATA_DIR} --net-dir {NET_DIR} --log-dir {LOG_DIR} --params {JSON_FILE} --id {MODEL_ID}
```

To print the training logs for any given step run

```
python3 results.py
```
'stage 4' corresponds to the best models chosen and evaluated on the test set for the final results.


To evaluate a model on the test set do

```
python3 test.py --data-dir {DATA_DIR} --checkpoint {PTH.TAR FILE} --log-dir {LOG_DIR} --params {JSON_FILE}
```

#### Directory and additional files
- `models/`
	- `dataloader.py`
	- 

## Results

| RGB | NIR | Ground Truth |
| :---: | :---: | :---: |
| ![RGB](./figures/20250818T185831_20250818T185825_T10SEH_R113_33_15_B0X_rgb.tif) | ![Near-infrared](./figures/20250818T185831_20250818T185825_T10SEH_R113_33_15_B0X_nir.tif) | ![Label](./figures/20250818T185831_20250818T185825_T10SEH_R113_33_15_LBL.tif) |

![A chip histogram](./figures/20250818T185831_20250818T185825_T10SEH_R113_33_15_hist.png)

The dataset consists of 19 MGRS tiles distributed over the western US. Two tiles of the original 21 tiles were arbitrarily dropped due to substantial overlap with tiles in an adjacent UTM zone (see red tiles below left). For the train/val/test split, 3 MGRS tiles were assigned to testing, and three to validation (see blue,red tiles below).

| Tiles Dropped  | Train/val/test (green/red/blue) |
| :---: | :---: |
| ![Bad tiles](./figures/tiles.png) | ![Split](./figures/tiles_split.png) |

All weeks produced rasters, with most images being retrieved during the summer (lower cloud coverage). Most rasters produced 0-50 chips. Most windows produced between 0-10 chips, and the average revisit time for a window (i.e. to produce a chip) was ~9.8 days.
![Chips per week](./figures/chips_per_week.png)
<!--![Chips per month](./figures/chips_per_month.png)-->


| Chips per raster | Chips per window | Revisit time |
| :---: | :---: | :---: |
| ![RGB](./figures/chips_per_raster.png) | ![Near-infrared](./figures/chips_per_window.png) | ![Label](./figures/window_revisit.png) |