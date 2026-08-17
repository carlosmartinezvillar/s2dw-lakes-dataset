'''
Sets experiments.
Each experiment is a row with a dictionary of hyperparameters.
'''
import itertools
import json
import argparse
import os
import numpy as np
import random
import copy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def stage_1():
	# STAGE 1 -- GET BEST LEARNING RATE, BATCH SIZE, WEIGHT DECAY

	# FIXED PARAMETERS
	name = 'stage_1'
	n_trials = 40
	models   = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	rows = []
	for i,m in enumerate(models):

		# SEARCH
		lrate = 10**np.random.uniform(-5,-2,size=n_trials)
		decay = 10**np.random.uniform(-4,-2,size=n_trials)
		batch = np.random.choice([8,16],size=n_trials)

		for j in range(n_trials):
			model_id = i*n_trials+j
			sample = {
				'id':model_id,
				'model':m,
				'seed':476,
				'epochs':35,
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
				'mlp_ratio':5, #base
				'cnn_layers':2, #base
				'channels':32   #base
			}

			rows.append(sample)

	# write to ./hparams/stage_1.json
	write_hp_file(name,rows)


def stage_2():
	'''
	STAGE 2 -- BEST LEARNING RATE, WEIGHT DECAY ACROSS SEEDS
	'''

	# FIXED PARAMETERS
	name = 'stage_2'
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
	best  = [15,13,19,76,70,67,101,90,84,153,141,138] #best 3 per model from stage_1()

	# SEARCH
	seeds = [176,276,376,476,576]	
				 
	with open('./hparams/stage_1.json','r') as fp:
		stage_1 = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_stage_1 = {row['id']:row for row in stage_1}

	# COMBINE
	cross_product = list(itertools.product(seeds,best))

	rows   = []
	for i,(s,idx) in enumerate(cross_product):
		sample = copy.deepcopy(indexed_stage_1[idx])
		sample["seed"]   = s
		sample["epochs"] = 65
		sample["old_id"] = sample["id"]
		sample["id"]     = i
		rows.append(sample)

	# write to ./hparams/stage_2.json
	write_hp_file(name,rows)


def stage_3():
	'''
	STAGE 3 -- CHECK SCHEDULER
	'''

	# FIXED PARAMETERS
	name = 'stage_3'

	#BEST MODELS FOR EACH IN STAGE 1 & 2: 19,76,84,138 
	# 19: "lrate": 0.00022, "decay": 0.00837
	# 76: "lrate": 0.0002, "decay": 0.00055
	# 84: "lrate": 0.0002, "decay": 0.00786
	# 138: "lrate": 0.00019, "decay": 0.00114
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
	bestlr = [0.00022,0.0002,0.0002,0.00019]
	bestwd = [0.00837,0.00055,0.00786,0.00114]

	# SEARCH
	all_eta_min = [0.0,1e-6,1e-5]
	all_cycles  = [1,2,3]

	# COMBINE
	cross_product = list(itertools.product(models,all_eta_min,all_cycles))

	rows = []
	for i,combination in enumerate(cross_product):

		model   = combination[0]
		eta_min = combination[1]
		cycles  = combination[2]

		lr = bestlr[models.index(model)]
		wd = bestwd[models.index(model)]

		sample = {
			'id':i,
			'model':model,
			'seed':476,
			'epochs':65,
			'scheduler':"cos",
			'eta_min':eta_min,
			'cycles': cycles,
			'loss':"ce",
			'bands':3,
			'labels':2,
			'optim':"adamw",
			'lrate':lr,
			'decay':wd,
			'batch':16,
			'vit_layers':1,
			'mlp_ratios':5,
			'cnn_layers':2,
			'channels':32
		}
		rows.append(sample)

	# write to ./hparams/stage_3.json
	write_hp_file(name,rows)


def stage_4():
	'''
	STAGE 4 -- PARAMETER SIZE: CHANNELS x DEPTH SIZE
	variations = [UNet_CNN_CNN,UNet_ViT_CNN,UNet_CNN_ViT,UNet_ViT_ViT]
	tiny  = {'cnn_layers':2,'vit_layers':1,'channels':16,'mlp_ratio':5}
	small = {'cnn_layers':3,'vit_layers':2,'channels':16,'mlp_ratio':5}
	base  = {'cnn_layers':2,'vit_layers':1,'channels':32,'mlp_ratio':5}
	large = {'cnn_layers':3,'vit_layers':2,'channels':32,'mlp_ratio':5}
	'''	

	# FIXED PARAMETERS
	name = 'stage_4'
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]
	bestlr = [0,0,0,0] #missing
	bestwd = [1,1,1,1]
	best_eta_min = [0,0,0,0]
	best_cycles  = [1,1,1,1]	

	# SEARCH
	cnn_layers = [2,3] # follows vit_layers = [1,2]
	channels   = [16,32]
	
	# COMBINE
	cross_product = list(itertools.product(models,channels,cnn_layers)) #16 combinations

	rows = []
	for i,(model,channels,cnn_layers) in enumerate(cross_product):

		if cnn_layers == 3:
			vit_layers = 2
		else:
			vit_layers = 1

		lr = bestlr[models.index(model)]
		wd = bestwd[models.index(model)]
		s_min = best_eta_min[models.index(model)]
		s_cyc = best_cycles[models.index(model)]

		sample = {
			'id':i,
			'model':model,
			'seed':476,
			'epochs':65,
			'scheduler':"cos",
			'eta_min': s_min,
			'cycles': s_cyc,
			'loss':"ce",
			'bands':3,
			'labels':2,
			'optim':"adamw",
			'lrate':lr,
			'decay':wd,
			'batch':16,
			'vit_layers':vit_layers,
			'mlp_ratio':5,
			'cnn_layers':cnn_layers, 
			'channels':channels  
		}
		rows.append(sample)

	write_hp_file(name,rows)


def stage_5():
	'''
	STAGE 5 -- CNN STEM+PATCHING VS PREVIOUS <<<- add CNN2 !
	'''

	# FIXED PARAMETERS
	name = 'stage_5'
	bestlr = [0,0,0,0] #missing
	bestwd = [1,1,1,1]
	best_eta_min = [0,0,0,0]
	best_cycles  = [1,1,1,1]

	# SEARCH
	models = ["UNet_ViT2_CNN","UNet_ViT2_ViT","UNet_CNN2_CNN","UNet_CNN2_ViT"]

	rows = []
	for i,m in enumerate(models):

		lr = bestlr[models.index(m)]
		wd = bestwd[models.index(m)]
		s_min = best_eta_min[models.index(m)]
		s_cyc = best_cycles[models.index(m)]

		sample = {
			'id':i,
			'model':m,
			'seed':476,
			'epochs':65,
			'scheduler':"cos",
			'eta_min': s_min,
			'cycles': s_cyc,
			'loss':"ce",
			'bands':3,
			'labels':2,
			'optim':"adamw",
			'lrate':lr,
			'decay':wd,
			'batch':16,
			'vit_layers':1,
			'mlp_ratio':5,
			'cnn_layers':2, 
			'channels':16 
		}
		rows.append(sample)

	write_hp_file(name,rows)


def write_hp_file(name,rows):
	# WRITE JSON FILE
	out_file_path = f"./hparams/{name}.json"		
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':
	set_seed(476) #Set seed to fix list of hyperparameters
	# stage_1()
	# stage_2()
	stage_3()
	# stage_4()
	