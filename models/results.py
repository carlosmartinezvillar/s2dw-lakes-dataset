'''
Retrieve training results from training logs.
This script assumes log files are stored as:

/LOG_DIR/stage_1/epochs_000.tsv
/LOG_DIR/stage_1/epochs_001.tsv
...
/LOG_DIR/stage_2/epochs_000.tsv
...etc. 
'''
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import argparse
import json

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

	assert len(lines) > 1, f"Found {len(lines)} in {log_path}"

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


def plot_training_log(log_path,best_iou_epoch=None,best_ema_epoch=None):
	'''
	Plot full time series of per epoch training and validation results.
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

	if best_iou_epoch is not None:
		ax.axvline(x=best_iou_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	out_path_1 = f'../figures/loss_{stage_nr}_{model_id}.png'
	plt.savefig(out_path_1)
	plt.close()
	print(f"Plot written to {out_path_1}")

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

	if best_iou_epoch is not None:
		ax.axvline(x=best_iou_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	out_path_2 = f'../figures/metrics_{stage_nr}_{model_id}.png'
	plt.savefig(out_path_2)
	plt.close()
	print(f"Plot written to {out_path_2}")


def sort_ids_by_model(models,hp_list):
	model_id_dict = {key:[] for key in models}
	for row in hp_list:
		model_id_dict[row['model']].append(row['id'])	
	return model_id_dict


def plot_lrate_vs_decay(model_str,lrates,decays,scores):
	'''
	For 'stage_1'.
	Plot a scatter plot for the decay and lrate hyperparameter search
	of a model.
	'''

	lrates = np.array(lrates)
	decays = np.array(decays)
	scores = np.array(scores)


	# Indices of the top 5 scores
	top5_idx = np.argsort(scores)[-5:]
	mask = np.zeros(len(scores), dtype=bool)
	mask[top5_idx] = True

	# exp_ids = np.array(exp_ids) # debugging
	# print(scores[top5_idx])
	# print(exp_ids[top5_idx])

	out_path = f'../figures/decaylrate_{model_str}.png'

	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)

	# Plot the rest as dots
	ax.scatter(
		lrates[~mask], decays[~mask],
		marker='o',
		edgecolors='white',
		linewidths=0.5
	)

	# Plot the top 5 as 'x'
	ax.scatter(
		lrates[mask], decays[mask],
		marker='x',
		color='red',
		linewidths=1.5,
		s=100,
		label='Top 5'
	)

	# adjust plot
	ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
	ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
	ax.set_ylabel('Decay')
	ax.set_xlabel('Learning Rate')
	ax.set_title(f"Decay vs. Learning Rate -- {model_str}; N={len(scores)}")
	plt.savefig(out_path)
	plt.close()

	print(f"Plot written to {out_path}")


def plot_batch_vs_iou(model_str,model_scores,model_batches,ema=False):
	'''
	For 'stage 1'.
	Boxplot for distribution of IoU for each batch size.
	'''
	# EASIER TYPE
	model_scores = np.array(model_scores)
	model_batches = np.array(model_batches)

	# GROUP BY BATCH SIZE
	unique_batches = np.unique(model_batches) #8,16
	grouped_scores = [model_scores[model_batches==b] for b in unique_batches]
	group_labels   = [f"{b}" for b in unique_batches]

	# PLOT
	out_path = f'../figures/batchiou_{model_str}.png'
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	ax.boxplot(grouped_scores,labels=group_labels)

	# ADJUST PLOT
	if ema:
		ax.set_ylabel('Validation IoU -- EMA')
	else:
		ax.set_ylabel('Validation IoU')
	ax.set_xlabel('Batch Size')
	ax.set_title(f"Validation IoU by Batch Size-- {model_str}")
	plt.savefig(out_path)
	plt.close()	
	print(f"Plot written to {out_path}")


def check_log_dir(log_dir,folder_range=160):
	expected_files = {f"epochs_{i:03}.tsv" for i in range(folder_range)}
	present_files = set(glob.glob("epochs_*.tsv",root_dir=log_dir))
	missing_files = sorted(expected_files - present_files)
	print(missing_files)


def get_best_stage_1(log_dir):
	'''
	Get best lrate, batch, decay for each model variation.
	'''

	# --------------------------------------------------
	# LOAD & SET STRINGS
	# --------------------------------------------------
	with open('./hparams/stage_1.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	# --------------------------------------------------
	# GROUP ROW/EXPERIMENTS BY MODEL
	# --------------------------------------------------
	# {'UNet_CNN_CNN':[0,1,2,...],'UNet_ViT_CNN':[40,41,42,..], ...etc.}
	ids_by_model = sort_ids_by_model(models,hp_list)

	# --------------------------------------------------
	# GET BEST EPOCH RESULTS FOR EACH EXPERIMENT
	# --------------------------------------------------
	'''
	{'UNet_CNN_CNN':
	[
		{
			'id': '000',
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
			try:
				model_results[model].append(get_model_best_epoch(log_file))
			except AssertionError as e:
				print(f"Error loading file: {e}")

	# --------------------------------------------------
	# FILTER BEST 5 BY IOU (or EMA IoU?)
	# --------------------------------------------------
	'''
	best_by_model = {
		'UNet_CNN_CNN': [(model_id,(iou,epoch))]	
		...etc.
	}
	'''
	best_by_model = {key:[] for key in models}
	for model in model_results:
		scores = model_results[model]
		ious = [_['iou'] for _ in scores]
		emas = [_['ema'] for _ in scores]
		top5 = sorted(enumerate(emas),key=lambda x: x[1],reverse=True)[:5] #[(i,(score,epoch))]
		top5_idx = [_[0] for _ in top5]
		top5_ema = [_[1][0] for _ in top5]
		top5_epo = [_[1][1] for _ in top5]

		for idx in top5_idx:
			best_by_model[model].append(scores[idx])

	# --------------------------------------------------
	# STDOUT/TXT BEST RUNS PER ARCHITECTURE
	# --------------------------------------------------
	indexed_hp_list = {row['id']:row for row in hp_list}
	best_hp_ids = []
	fp =  open('./hparams/best_stage_1.txt','w')
	for model in best_by_model:
		print(f"\n{model} -- TOP 5 SCORES")
		print('-'*20)
		for score_dict in best_by_model[model]:
			hp_dict = indexed_hp_list[int(score_dict['id'])]
			line = f"id: {score_dict['id']} | iou: {score_dict['iou']} | ema: {score_dict['ema']}"
			line += f" | batch: {hp_dict['batch']} | lrate: {hp_dict['lrate']} | decay: {hp_dict['decay']}"
			print(line)
			fp.write(line + '\n')
			best_hp_ids.append(int(score_dict['id']))
	fp.close()

	return

	# --------------------------------------------------
	# SAVE A NEW FILE WITH HPARAMS SET FOR THESE BEST 5
	# --------------------------------------------------
	# indexed_hp_list = {row['id']:row for row in hp_list}
	rows = [indexed_hp_list[i] for i in best_hp_ids]
	out_file_path = f"./hparams/best_stage_1.json"
	with open(out_file_path,'w') as fp:
		for row in rows:
			json.dump(row,fp)
			fp.write('\n')
	print(f"\nParameter file written to {out_file_path}")


	# --------------------------------------------------
	# PLOT TRAINING LOG BEST 5
	# --------------------------------------------------
	for model in best_by_model:
		for score_dict in best_by_model[model]:
			experiment = score_dict['id']
			log_file   = f"{log_dir}/stage_1/epochs_{experiment:03}.tsv"
			plot_training_log(log_file)

	# --------------------------------------------------
	# MATCH LRATE, BATCH, & DECAY TO SCORE
	# --------------------------------------------------
	# score_and_config = {k:[] for k in model_results.keys()}
	for model in model_results:
		model_ids    = [] #debugging
		model_lrates = []
		model_decays = []
		model_emas   = []
		model_batches = []
		model_ious   = []
		scores = model_results[model]
		for score_dict in scores:
			score_id    = int(score_dict['id'])
			score_lrate = indexed_hp_list[score_id]['lrate']
			score_decay = indexed_hp_list[score_id]['decay']
			score_batch = indexed_hp_list[score_id]['batch']
			score_iou   = score_dict['iou'][0]
			score_ema   = score_dict['ema'][0]
			# score_and_config[model].append((score_id,score_lrate,score_decay,score_batch,score_iou,score_ema))
			model_lrates.append(score_lrate)
			model_decays.append(score_decay)
			model_emas.append(score_ema)
			model_ious.append(score_iou)
			model_batches.append(score_batch)
			model_ids.append(score_id)

		# --------------------------------------------------
		# PLOT -- EACH MODEL DIST. OF LRATE DECAY
		# --------------------------------------------------
		plot_lrate_vs_decay(model,model_lrates,model_decays,model_emas)

		# --------------------------------------------------
		# PLOT -- EACH MODEL BOXPLOT, BATCH vs IoU
		# --------------------------------------------------
		plot_batch_vs_iou(model,model_emas,model_batches)


def get_best_stage_2(log_dir):
	'''
	Run through all 16 combinations in Stage 2.
	'''
	# --------------------------------------------------
	# LOAD & SET STRINGS
	# --------------------------------------------------
	with open('./hparams/stage_2.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	# --------------------------------------------------
	# VALIDATION RESULTS
	# --------------------------------------------------
	all_results = []
	for row in hp_list:
		experiment = row['id']
		log_file   = f"{log_dir}/stage_2/epochs_{experiment:03}.tsv" # <--- fails if no log
		best_epoch = get_model_best_epoch(log_file)
		all_results.append(best_epoch)

	for score_dict in all_results:
		line = f"id: {score_dict['id']} | iou: {score_dict['iou']} | ema: {score_dict['ema']}"
		print(line)

	# --------------------------------------------------
	# TEST RESULTS
	# --------------------------------------------------
	for i,row in enumerate(hp_list):
		experiment = row['id']
		log_file = f"{log_dir}/stage_2/test_{experiment:03}.tsv"
		with open(log_file,'r') as fp:
			lines = fp.readlines()
		header = lines[0]
		result = lines[1].rstrip('\n')
		if i == 0:
			print(f"{'-'*20} | {header}")
		line = f"{row['model']} | ID: {experiment} | {result}"
		print(line)


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

	# experiment = 0
	# log_file = f"{log_dir}/stage_1/epochs_{experiment:03}.tsv"

	# best = get_model_best_epoch(log_file)
	# print(best)
	# best_ema_epoch = best['ema'][1]
	# plot_training_log(log_file,best_epoch=best_ema_epoch)

	# check_log_dir(log_dir)
	get_best_stage_1(log_dir)