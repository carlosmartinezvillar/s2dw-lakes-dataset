'''
Set files with rows of hyperparameter dictionaries.
'''
import itertools
import json
import argparse
import os
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_hyperparameters(name):
	'''
	Set combinations of hyperparameters.
	'''
	# LEARNING RATE, BATCH SIZE, WEIGHT DECAY
	if name == 'stage_1':
		n_trials = 30
		seed       = 476
		epochs     = 25
		scheduler  = "cos"
		loss_func  = "ce"
		bands      = 3
		labels     = 2
		vit_layers = 1 #base
		cnn_layers = 2 #base
		channels   = 32 #base
		model      = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

		rows = []

		for i,m in enumerate(models):

			lrate = 10**np.random.uniform(-5,-2,size=n_trials)
			decay = 10**np.random.uniform(-4,-2,size=n_trials)
			batch = np.random.choice([8,16],size=n_trials)

			for j in range(n_trials):
				model_id = i*n_trials+j
				sample = {
					'id':model_id,
					'model':m,
					'seed':476,
					'epochs':25,
					'scheduler':"cos",
					'loss':"ce",
					'bands':3,
					'labels':2,
					'vit_layers':1, #base
					'cnn_layers':2, #base
					'channels':32   #base
				}

				rows.append(sample)


	# ARCHITECTURE VARIATIONS: CHANNELS x DEPTH SIZE
	if name == 'stage_2':
		seed   = 476
		epochs = 50
		lrate = [] #best from 'learning_rates'
		decay = [] #best from 'learning_rates'

		models ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
		vit_layers = [1,2]
		# cnn_layers = [2,3]
		channels   = [16,32]
		# mlp_ratios = [2,4]

		# Cross-product
		cross_product = list(itertools.product())
		HP_NEW = []

		for i in range(len(hp)):
			row_dict = {}
			row_dict["ID"]            = i
			row_dict["SEED"]          = hp[i][0]
			row_dict["EPOCHS"]        = hp[i][1]		
			row_dict["LEARNING_RATE"] = hp[i][2]
			row_dict["SCHEDULER"]     = hp[i][3]
			row_dict["OPTIM"]         = hp[i][4]
			row_dict["DECAY"]         = hp[i][5]
			row_dict["LOSS"]          = hp[i][6]
			row_dict["BATCH"]         = hp[i][7]
			row_dict["INIT"]          = hp[i][8]
			row_dict["BANDS"]         = hp[i][9]
			row_dict["OUTPUTS"]       = hp[i][10]
			row_dict["MODEL"]         = hp[i][11]
			HP_NEW.append(row_dict)


	if name == 'stage_3':
		# eta_min: [0.0,1e-6,1e-5]
		# cycles: [1,2,3]
		pass


	out_file_path = f"./hparams/{name}.json"		
	assert not os.path.isfile(out_file_path), f"Overwriting existing file {out_file_path}"
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':

	set_seed(476)
	# exp2 = 'stem_check'

	set_hyperparameters('stage_1')
	# sequence_hyperparameters(out_file_path,id_start=101,trial=1)		
	# sequence_hyperparameters(out_file_path,id_start=421,trial=2)
