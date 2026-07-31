'''
Retrieve training results from training logs.

Assumes log files are stored as:

/LOG_DIR/stage_1/epochs_000.tsv
/LOG_DIR/stage_1/epochs_001.tsv
...
/LOG_DIR/stage_2/epochs_000.tsv etc. 
'''

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse

def calculate_ema(metric,alpha=0.3):
	'''
	Return EMA for np.array 'metric'.
	'''
	ema = np.zeros_like(metric)
	ema[0] = metric[0] # first EMA value
	for i in range(1, len(metric)): # Calculate rest of the array
	    value = alpha * metric[i] + (1 - alpha) * ema[i - 1]
	    ema[i] = round(value,5)
	return ema


def load_log(log_path):
	# OPEN/READ
	with open(log_path,'r') as fp:
		lines = fp.readlines()
	header = lines[0].rstrip('\n').split('\t')
	epochs = np.array([l.rstrip('\n').split('\t') for l in lines[1:]]).astype(float)

	# RETURN
	return header, epochs	


def get_model_best_epoch(log_path):
	'''
	Read a log file. 1st line header. Each line is an epoch.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET ID
	model_id = log_path.rstrip('.tsv').split('_')[-1]

	# LOAD
	header,epochs = load_log(log_path)

	# GET VALIDATION COLUMNS
	iou_idx = header.index('viou1')
	acc_idx = header.index('vacc1')
	tpr_idx = header.index('vtpr1')
	ppv_idx = header.index('vppv1')

	# GET EMA of IoU
	ema = calculate_ema(epochs[:,iou_idx])

	# GET MAX VALUE & MAX INDEX
	best_ema = np.max(ema)
	best_iou = np.max(epochs[:,iou_idx])
	best_acc = np.max(epochs[:,acc_idx])
	best_tpr = np.max(epochs[:,tpr_idx])
	best_ppv = np.max(epochs[:,ppv_idx])
	best_ema_epoch = np.argmax(ema)
	best_iou_epoch = np.argmax(epochs[:,iou_idx])
	best_acc_epoch = np.argmax(epochs[:,acc_idx])
	best_tpr_epoch = np.argmax(epochs[:,tpr_idx])
	best_ppv_epoch = np.argmax(epochs[:,ppv_idx])

	best = {
		'id': model_id,
		'iou':(best_iou,best_iou_epoch),
		'acc':(best_acc,best_acc_epoch),
		'tpr':(best_tpr,best_tpr_epoch),
		'ppv':(best_ppv,best_ppv_epoch),
		'ema':(best_ema,best_ema_epoch)
	}
	return best  


def plot_training_log(log_path,best_epoch=None):
	'''
	Plot full time series of epochs log for training and validation results.
	Two plots: loss and metrics.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET IDs
	model_id = log_path.rstrip('.tsv').split('_')[-1]
	stage_nr = log_path.split('/')[-2]

	# OPEN/READ
	header,epochs = load_log(log_path)

	# GET LOSS COLS
	tloss_idx = header.index('tloss')
	vloss_idx = header.index('vloss')
	tloss = epochs[:,tloss_idx]
	vloss = epochs[:,vloss_idx]

	# GET TRAIN METRIC COLS
	tiou1_idx = header.index('tiou1')
	tacc1_idx = header.index('tacc1')
	ttpr1_idx = header.index('ttpr1')
	tppv1_idx = header.index('tppv1')
	tiou1 = epochs[:,tiou1_idx]
	tacc1 = epochs[:,tacc1_idx]
	ttpr1 = epochs[:,ttpr1_idx]
	tppv1 = epochs[:,tppv1_idx]

	# GET VAL METRIC COLS
	viou1_idx = header.index('viou1')
	vacc1_idx = header.index('vacc1')
	vtpr1_idx = header.index('vtpr1')
	vppv1_idx = header.index('vppv1')
	viou1 = epochs[:,viou1_idx]
	vacc1 = epochs[:,vacc1_idx]
	vtpr1 = epochs[:,vtpr1_idx]
	vppv1 = epochs[:,vppv1_idx]

	####################
	# I. PLOT -- LOSS
	####################
	# SET
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':1.0}
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')
	ax.set_title(f"Training & Validation Loss (Model {model_id})")

	# PLOT
	ax.plot(tloss,label='Training',linestyle='--',**params)
	ax.plot(vloss,label='Validation',linestyle='-',**params)

	if best_epoch is not None:
		ax.axvline(x=best_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	plt.savefig(f'../figures/loss_stage{stage_nr}_{model_id}.png')
	plt.close()

	####################
	# II. PLOT - METRICS
	####################
	# CONFIG
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':1.0}
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Training & Validation Metrics")

	# PLOT
	ax.plot(tacc1,label='Train acc',linestyle='-.',**params)
	ax.plot(tiou1,label='Train IoU',linestyle='-.',**params)
	# ax.plot(ttpr1,label='Train tpr',linestyle='-.',**params)
	# ax.plot(tppv1,label='Train ppv',linestyle='-.',**params)
	# ax.plot(vacc1,label='Valid acc',linestyle='-',**params)
	ax.plot(viou1,label='Valid IoU',linestyle='-',**params)
	# ax.plot(vtpr1,label='Valid tpr',linestyle='-',**params)
	# ax.plot(vppv1,label='Valid ppv',linestyle='-',**params)

	if best_epoch is not None:
		ax.axvline(x=best_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	plt.savefig(f'../figures/metrics_stage{stage_nr}_{model_id}.png')
	plt.close()


def sort_ids_by_model(models,hp_list):
	model_id_dict = {key:[] for key in models}
	for row in hp_list:
		model_id_dict[row['model']].append(row['id'])	
	return model_id_dict


def plot_lrate_vs_decay(hp_list,ids,model_str):
	'''
	Plot a scatter plot for the decay and lrate hyperparameter search
	of a model.
	'''

	indexed = {row['id']:row for row in hp_list}

	decays = []
	lrates = []
	for i in ids:
		decays.append(indexed[i]['decay'])
		lrates.append(indexed[i]['lrate'])
	decays = np.array(decays)
	lrates = np.array(lrates)

	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	# params = {'linewidth':0.8}
	# ax.plot(vppv1,label='Valid ppv',linestyle=':',**params)
	ax.scatter(
	lrates, decay, 
	# s=sizes, 
	# c=colors, 
	cmap='plasma', 
	# alpha=0.8,
	edgecolors='white',
	linewidths=0.5
	)
	ax.set_ylabel('Decay')
	ax.set_xlabel('LRate')
	ax.set_title(f"Decay vs. Learning Rate -- {model_str}")
	plt.savefig(f'../figures/decaylrate_{model_str}.png')
	plt.close()


def plot_batch_vs_iou():
	'''
	Needed only(?) for stage 1.
	Histogram or boxplot for distribution of IoU for each batch size.
	'''
	pass
	# ax.set_ylabel('Validation IoU')
	# ax.set_xlabel('Batch Size')
	# ax.set_title(f"Batch vs. Validation IoU -- {model_str}")
	# plt.savefig(f'../figures/batch_{model_str}.png')
	# plt.close()	


def get_best_stage_1(log_dir):
	'''
	Get best lrate, batch, decay for each model variation.
	'''

	with open('./hparams/stage_1.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]

	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	# ROW/EXPERIMENT GROUPED BY MODEL
	# {'UNet_CNN_CNN':[0,1,2,...],'UNet_ViT_CNN':[40,41,42,..], ...etc.}
	ids_by_model = sort_ids_by_model(models,hp_list)


	# GET BEST EPOCH RESULTS FOR EACH EXPERIMENT
	'''
	{'UNet_CNN_CNN':
	[
		{
			'id': 000,
			'iou':(best_iou,best_iou_epoch),
			'acc':(best_acc,best_acc_epoch),
			'tpr':(best_tpr,best_tpr_epoch),
			'ppv':(best_ppv,best_ppv_epoch),
			'ema':(best_ema,best_ema_epoch)
		},
		...
	],
	'UNet_ViT_CNN':[], ...etc}
	'''
	model_results = {key:[] for key in models}
	for model in model_results:
		for experiment in ids_by_model[model]:
			log_file = f"{log_dir}/stage_1/epochs_{experiment:03}.tsv" # <--- fails if no log
			model_results[model].append(get_model_best_epoch(log_file))


	# FIND BEST 5 BY IOU (EMA IoU?)
	'''
	best_by_model = {
		'UNet_CNN_CNN': [(model_id,(iou,epoch))]	
		...
	}
	'''
	best_by_model = {key:[] for key in models}
	for model in model_results:
		scores = model_results[model]
		ious = [_['iou'] for _ in scores]
		emas = [_['ema'] for _ in scores]
		top5 = sorted(enumerate(ious),key=lambda x: x[1],reverse=True)[:5] #[(i,(score,epoch))]
		top5_idx = [_[0] for _ in top5]
		top5_iou = [_[1][0] for _ in top5]
		top5_epo = [_[1][1] for _ in top5]

		for idx in top5_idx:
			best_by_model[model].append(scores[idx])

	# STDOUT BEST RUNS PER ARCHITECTURE
	for model in best_by_model:
		print(f"\n{model} -- TOP 5 SCORES")
		print('-'*20)
		for score_dict in best_by_model[model]:
			print(f"id: {score_dict['id']} | iou: {score_dict['iou']} | ema: {score_dict['ema']}")


	# PLOT TRAINING LOG BEST 5
	for model in best_by_model:
		for score_dict in best_by_model[model]:
			exp = score_dict['id']
			log_file = f"{log_dir}/stage_1/epochs_{exp:03}.tsv"
			plot_training_log(log_file)



	# MATCH LRATE & DECAY TO SCORE
	hp_indexed = {row['id']:row for row in hp_list}
	score_and_config = {k:[] for k in model_results.keys()}
	for model in model_results:
		scores = model_results[model]
		for score_dict in scores:
			score_id    = int(score_dict['id'])
			score_lrate = hp_indexed[score_id]['lrate']
			score_decay = hp_indexed[score_id]['decay']
			# score_iou   = score_dict['iou'][0]
			score_ema   = score_dict['ema'][0]
			score_and_config[model].append((score_id,score_lrate,score_decay,score_ema))

		# PLOT THE PREVIOUS WITH SCORE HIGHLIGHTED
		plot_lrate_vs_decay(,model)


def get_best_stage_2():
	'''
	Run through all 16 combinations in Stage 2.
	'''
	pass


def get_best_stage_3():
	'''
	Get the best cosine scheduler parameters
	'''
	pass


def get_best_stage_4():
	'''
	Evaluate the training performance of ViT2 (patch embedding).
	'''
	pass


def parse_args():
	# ARGV
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--log-dir',required=True,help='Training logs.')

	# LOAD 
	args = parser.parse_args()

	# CHECK HERE
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	return args	


if __name__ == '__main__':
	args = parse_args()
	log_dir = args.log_dir.rstrip('/')

	experiment = 0
	log_file = f"{log_dir}/stage_1/epochs_{experiment:03}.tsv"

	best = get_model_best_epoch(log_file)
	print(best)
	best_ema_epoch = best['ema'][1]
	plot_training_log(log_file,best_epoch=best_ema_epoch)


