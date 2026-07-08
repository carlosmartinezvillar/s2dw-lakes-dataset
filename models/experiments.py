'''
Set files with rows of hyperparameter dictionaries.
'''
import itertools
import json
import argparse
import os
import numpy as np
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_hyperparameters(name):
	'''
	Set combinations of hyperparameters.
	'''
	# STAGE 1 -- LEARNING RATE, BATCH SIZE, WEIGHT DECAY
	if name == 'stage_1':
		n_trials = 40
		seed       = 476
		epochs     = 25
		scheduler  = "cos"
		loss_func  = "ce"
		bands      = 3
		labels     = 2
		vit_layers = 1 #base
		cnn_layers = 2 #base
		channels   = 32 #base
		mlp_ratios = 4
		models     = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

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
					'eta_min':0.0,
					'cycles': 1,
					'loss':"ce",
					'bands':3,
					'labels':2,
					'optim':"adamw",
					'lrate':round(lrate[j],5),
					'decay':round(decay[j],5),
					'batch':int(batch[j]),					
					'vit_layers':1, #base
					'mlp_ratios':4, #base
					'cnn_layers':2, #base
					'channels':32   #base
				}

				rows.append(sample)


	# STAGE 2 -- ARCHITECTURE VARIATIONS: CHANNELS x DEPTH SIZE
	if name == 'stage_2':
		seed   = 476
		epochs = 50
		scheduler  = "cos"
		loss_func  = "ce"
		bands      = 3
		labels     = 2

		lrate = [] #best from stage 1
		decay = [] #best from stage 1
		batch = [] #best from stage 1

		models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
		cnn_layers = [2,3] # follows vit_layers = [1,2]
		channels   = [16,32] # follows mlp_ratios = [2,4]
		
		# Cross-product
		cross_product = list(itertools.product(models,channels,cnn_layers))
		rows = []

		for i in range(len(cross_product)):
			model = cross_product[i][0]
			channels = cross_product[i][1]
			cnn_layers = cross_product[i][2]

			if channels == 32:
				mlp_ratios = 4
			else:
				mlp_ratios = 2

			if cnn_layers == 3:
				vit_layers = 2
			else:
				vit_layers = 1

			sample = {
				'id':i,
				'model':model,
				'seed':476,
				'epochs':50,
				'scheduler':"cos",
				'eta_min':0.0,
				'cycles': 1,
				'loss':"ce",
				'bands':3,
				'labels':2,
				'optim':"adamw",
				'lrate':0, #missing
				'decay':0, #missing
				'batch':8, #missing
				'vit_layers':vit_layers,
				'mlp_ratios':mlp_ratios,
				'cnn_layers':cnn_layers, 
				'channels':channels  
			}
			rows.append(sample)


	# STAGE 3 -- CHECK SCHEDULER
	if name == 'stage_3':
		models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
		all_eta_min = [0.0,1e-6,1e-5]
		all_cycles  = [1,2,3]

		# Cross-product
		cross_product = list(itertools.product(models,all_eta_min,all_cycles))


		rows = []
		for i,combination in enumerate(cross_product):

			model   = combination[0]
			eta_min = combination[1]
			cycles  = combination[2]

			sample = {
				'id':i,
				'model':model,
				'seed':476,
				'epochs':50,
				'scheduler':"cos",
				'eta_min':eta_min,
				'cycles': cycles,
				'loss':"ce",
				'bands':3,
				'labels':2,
				'optim':"adamw",
				'lrate':0, #missing -- stage 1
				'decay':0, #missing
				'batch':8, #missing
				'vit_layers':1, #missing -- stage 2
				'mlp_ratios':4, #missing
				'cnn_layers':2,  #missing
				'channels':32 #missing  
			}
			rows.append(sample)


	# STAGE 4 -- CNN STEM+PATCHING VS PREVIOUS
	if name == 'stage_4':
		models = ["UNet_ViT2_CNN","UNet_ViT2_ViT"]

		rows = []

		for i,m in enumerate(models):
			sample = {
				'id':i,
				'model':m,
				'seed':476,
				'epochs':50,
				'scheduler':"cos",
				'eta_min':0.0, #missing -- stage 3
				'cycles': 1, #missing -- stage 3
				'loss':"ce",
				'bands':3,
				'labels':2,
				'optim':"adamw",
				'lrate':0, #missing -- stage 1
				'decay':0, #missing
				'batch':8, #missing
				'vit_layers':1, #missing -- stage 2
				'mlp_ratios':4,
				'cnn_layers':2, 
				'channels':32  
			}
			rows.append(sample)


	# WRITE FILE
	out_file_path = f"./hparams/{name}.json"		
	# assert not os.path.isfile(out_file_path), f"Overwriting existing file {out_file_path}"
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':
	set_seed(476)
	set_hyperparameters('stage_1')
	set_hyperparameters('stage_2')
	set_hyperparameters('stage_3')
	set_hyperparameters('stage_4')
