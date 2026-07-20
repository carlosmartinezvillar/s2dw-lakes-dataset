'''
Get some dataset metrics.
'''
from PIL import Image
import glob
import numpy as np


def get_dataset_mean_std(dir_path):

	chip_ids = glob.glob(f"{dir_path}/*_B0X.tif")

	count = 0

	sum_   = np.zeros(4,dtype=np.float64)
	sum_sq = np.zeros(4,dtype=np.float64)


	for chip in chip_ids:

		bands = Image.open(chip).split()		
		arr = np.stack([np.array(b, dtype=np.float64) for b in bands])

		pixels = arr.reshape(4,-1)
		sum_ += pixels.sum(axis=1)
		sum_sq += (pixels ** 2).sum(axis=1)
		count  += pixels.shape[1]


	#MEAN & STD
	mean = sum_ / count
	std  = np.sqrt(sum_sq / count - mean ** 2)

	print("MEAN/STD for R,G,B,NIR:")
	print(mean)
	print(std)


def get_chip_positive_count():
	pass


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--chip-dir',required=True,help='Dataset directory.')
	args = parser.parse_args()

	if not os.path.isdir(args.chip_dir):
		print(f"Chip dir {args.chip_dir} not found. EXIT(1).")
		sys.exit(1)

	if args.chip_dir[-1] == '/':
		args.chip_dir = args.chip_dir[:-1]

	return args


if __name__ == '__main__':

	# ARGV/ARGUMENTS
	args = parse_args()
	data_dir = args.chip_dir

	# Get dataset mean/stddev
	get_dataset_mean_std(data_dir)