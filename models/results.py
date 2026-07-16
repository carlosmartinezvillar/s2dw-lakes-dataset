'''
Assumes results stored as:

/LOG_DIR/stage_1/epochs_000.tsv
/LOG_DIR/stage_1/epochs_001.tsv
...
/LOG_DIR/stage_2/epochs_000.tsv etc. 
'''

import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def get_model_results(log_path):
	'''
	Read a log file
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET ID
	model_id = log_path.rstrip('.tsv').split('_')[-1]

	# OPEN/READ
	with open(log_path,'r') as fp:
		lines = fp.readlines(log_path)
	header = lines[0].rstrip('\n').split('\t')
	epochs = np.array([l.rstrip('\n').split('\t') for l in lines[1:]]).astype(float)

	# GET VALIDATION COLUMNS
	iou_idx = header.index('viou1')
	acc_idx = header.index('vacc1')
	tpr_idx = header.index('vtpr1')
	ppv_idx = header.index('vppv1')

	# GET MAX VALUE & MAX INDEX
	best_iou = np.max(epochs[:,iou_idx])
	best_acc = np.max(epochs[:,acc_idx])
	best_tpr = np.max(epochs[:,tpr_idx])
	best_ppv = np.max(epochs[:,ppv_idx])
	best_iou_epoch = np.argmax(epochs[:,iou_idx])
	best_acc_epoch = np.argmax(epochs[:,acc_idx])
	best_tpr_epoch = np.argmax(epochs[:,tpr_idx])
	best_ppv_epoch = np.argmax(epochs[:,ppv_idx])

	best = {
		'id': model_id,
		'iou':(best_iou,best_iou_epoch),
		'acc':(best_acc,best_acc_epoch),
		'tpr':(best_tpr,best_tpr_epoch),
		'ppv':(best_ppv,best_ppv_epoch)
	}
	return best  


def plot_training_log(log_path):
	'''
	Plot the epochs/log for training and validation results.
	Two plots, loss and metrics.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET ID
	model_id = log_path.rstrip('.tsv').split('_')[-1]

	# OPEN/READ
	with open(log_path,'r') as fp:
		lines = fp.readlines(log_path)
	header = lines[0].rstrip('\n').split('\t')
	epochs = np.array([l.rstrip('\n').split('\t') for l in lines[1:]]).astype(float)

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
	tiou1 = epochs[:,tiou1]
	tacc1 = epochs[:,tacc1]
	ttpr1 = epochs[:,ttpr1]
	tppv1 = epochs[:,tppv1]

	# GET VAL METRIC COLS
	viou1_idx = header.index('viou1')
	vacc1_idx = header.index('vacc1')
	vtpr1_idx = header.index('vtpr1')
	vppv1_idx = header.index('vppv1')
	viou1 = epochs[:,viou1]
	vacc1 = epochs[:,vacc1]
	vtpr1 = epochs[:,vtpr1]
	vppv1 = epochs[:,vppv1]

	# PLOT -- LOSS
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':0.8}
	ax.plot(tloss,label='Training',linestyle='--',**params)
	ax.plot(vloss,label='Validation',linestyle='-',**params)
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')
	ax.set_title(f"Training & Validation Loss (Model {model_id})")
	plt.legend()
	plt.savefig(f'../figures/loss_{model_id}.png')
	plt.close()

	# PLOT - METRICS
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':0.8}
	ax.plot(tacc1,label='Train acc',linestyle='--',**params)
	ax.plot(tiou1,label='Train IoU',linestyle='-.',**params)
	ax.plot(ttpr1,label='Train tpr',linestyle='-.',**params)
	ax.plot(tppv1,label='Train ppv',linestyle='-.',**params)
	ax.plot(vacc1,label='Valid acc',linestyle=':',**params)
	ax.plot(viou1,label='Valid IoU',linestyle=':',**params)
	ax.plot(vtpr1,label='Valid tpr',linestyle=':',**params)
	ax.plot(vppv1,label='Valid ppv',linestyle=':',**params)
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Training & Validation Metrics")
	plt.legend()
	plt.savefig(f'../figures/metrics_{model_id}.png')
	plt.close()


def get_best_stage_1(log_dir):
	'''
	Get best lrate, batch, decay for each model variation.
	'''

	with open('./hparams/stage_1.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	# SET IDs AS KEYS and CHECK
	# hp_list_indexed = {row['id']:row for row in hp_list}

	models  =  ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	ids_by_model = {key:[] for key in models}
	for row in hp_list:
		ids_by_model[row['model']].append(row['id'])


	# GET BEST EPOCH RESULTS FOR EACH EXPERIMENT
	model_results = {key:[] for key in models}
	for model in model_results:
		for experiment in ids_by_model[model]:
			log_file = f"{log_dir}/stage_1/epochs_{experiment:03}.tsv"
			model_results[model].append(get_model_results(log_file))

	# FIND BEST 5 BY IOU
	best_by_model = {key:[] for key in models}
	for model in model_results:
		scores = model_results[model]
		ious = [_['iou'] for _ in scores]
		best = sorted(enumerate(ious),key=lambda x: x[1],reverse=True)[:5]
		best_by_model[model].append(best)
		# accs = []
		# tprs = []
		# ppvs = [] # no

	# PLOT TRAINING LOG BEST 5
	for model in best_by_model:
		for exp, max_iou in best_by_model[model]:
			log_file = f"{log_dir}/stage_1/epochs_{exp:03}.tsv"
			plot_training_log(log_file)


def get_best_stage_2():
	'''
	Get results of model size comparison for the 16 combinations
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


if __name__ == '__main__':
	pass

