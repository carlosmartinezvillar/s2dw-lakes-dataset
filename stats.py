'''
Get some dataset metrics.
'''

def get_dataset_mean_std():
	pass


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
	pass
	args = parse_args()