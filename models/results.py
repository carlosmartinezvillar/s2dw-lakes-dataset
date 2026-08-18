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
from scipy import stats
import itertools

FIG_SIZE = (10,5)


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


def load_train_log(log_path):
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
	Return a dictionary with tuple values, where tuples are
	a best metric value and its corresponding epoch.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET ID
	model_id = log_path.rstrip('.tsv').split('_')[-1]

	# LOAD
	header,epochs = load_train_log(log_path)

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
	header,epochs = load_train_log(log_path)

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
	fig = plt.figure(figsize=FIG_SIZE)
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
	out_path_1 = f'../figures/{stage_nr}/loss_{model_id}.png'
	plt.savefig(out_path_1)
	plt.close()
	print(f"Plot written to {out_path_1}")

	####################
	# II. PLOT - METRICS
	####################
	# CONFIG
	fig = plt.figure(figsize=FIG_SIZE)
	ax  = fig.add_subplot(111)
	params = {'linewidth':1.0}
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Training & Validation Metrics")

	# PLOT
	ax.plot(tacc1,label='Train acc',linestyle='-.',**params)
	ax.plot(tiou1,label='Train IoU',linestyle='-',**params)
	# ax.plot(ttpr1,label='Train tpr',linestyle='-.',**params)
	# ax.plot(tppv1,label='Train ppv',linestyle='-.',**params)
	ax.plot(vacc1,label='Valid acc',linestyle='-.',**params)
	ax.plot(viou1,label='Valid IoU',linestyle='-',**params)
	# ax.plot(vtpr1,label='Valid tpr',linestyle='-',**params)
	# ax.plot(vppv1,label='Valid ppv',linestyle='-',**params)

	if best_iou_epoch is not None:
		ax.axvline(x=best_iou_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	out_path_2 = f'../figures/{stage_nr}/metrics_{model_id}.png'
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
	Plot a scatter plot with decay and lrate for a model.
	'''

	# easier type
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

	out_path = f'../figures/stage_1/decaylrate_{model_str}.png'

	fig = plt.figure(figsize=FIG_SIZE)
	ax  = fig.add_subplot(111)

	norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
	cmap = plt.cm.plasma_r

	# Plot the rest as dots
	sc = ax.scatter(
		lrates[~mask], decays[~mask],
		c=scores[~mask],
		cmap=cmap,
		norm=norm,
		marker='o',
		edgecolors='black',
		linewidths=0.5
	)

	# Plot the top 5 as 'x'
	ax.scatter(
		lrates[mask], decays[mask],
		c=scores[mask],
		cmap=cmap,
		norm=norm,
		marker='x',
		# color='red',
		linewidths=2.0,
		s=150,
		label='Top 5'
	)

	cbar = fig.colorbar(sc, ax=ax)
	cbar.set_label('Max EMA(IoU)')

	# fixed axis ranges matching the min/max of the data
	ax.set_xlim(1e-5, 1e-2)
	ax.set_ylim(1e-4, 1e-2)  # NOTE: same value twice -- likely a typo, fix this

	# log scale since ranges span multiple orders of magnitude
	ax.set_xscale('log')
	ax.set_yscale('log')

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
	out_path = f'../figures/stage_1/batchiou_{model_str}.png'
	fig = plt.figure(figsize=FIG_SIZE)
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


def read_test_file(log_path):
	with open(log_path,'r') as fp:
		lines = fp.readlines()

	# READ TEST RESULTS
	header = lines[0]
	result = lines[1].rstrip('\n')
	test_iou = result[header.index('_iou1')]
	test_ppv = result[header.index('_ppv1')]
	test_tpr = result[header.index('_tpr1')]

	return test_iou,test_ppv,test_tpr


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
	os.makedirs("../figures/stage_1",exist_ok=True)
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
	Check a few (3) best parameter combinations in 'stage 1' for each model type
	over multiple seeds. 4 models x 3 combinations x 5 samples/seeds = 60 runs
	From here we select: 19, 76, 84, and 138
	'''
	with open('./hparams/stage_2.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]

	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	grouped_ids = {k:{} for k in models}
	for row in hp_list:
		if row['old_id'] not in grouped_ids[row['model']]:
			grouped_ids[row['model']].update({row['old_id']:[]})
		grouped_ids[row['model']][row['old_id']].append(row['id'])


	grouped_results = {
	    outer_key: {inner_key: None for inner_key in inner_dict}
	    for outer_key, inner_dict in grouped_ids.items()
	}

	# COLLECT VALIDATION RESULTS FOR ALL RUNS
	for a_model in grouped_ids:
		for old_id in grouped_ids[a_model]:
			ious = []
			emas = []
			for new_id in grouped_ids[a_model][old_id]:
				log_file = f"{log_dir}/stage_2/epochs_{new_id:03}.tsv"
				results  = get_model_best_epoch(log_file)
				ious.append(results['iou'][0])
				emas.append(results['ema'][0])

			grouped_results[a_model][old_id] = {'ious': np.array(ious),'emas': np.array(emas)}


	# PRINT STATS -- FOR EACH MODEL TYPE
	for a_model in grouped_results:
		print(f"\n{a_model}")
		print("-"*40)

		# AVGs PER MODEL TYPE
		for old_id in grouped_ids[a_model]:
			mean_iou = grouped_results[a_model][old_id]['ious'].mean().round(5)
			mean_ema = grouped_results[a_model][old_id]['emas'].mean().round(5)
			stdd_iou = grouped_results[a_model][old_id]['ious'].std().round(5)
			stdd_ema = grouped_results[a_model][old_id]['emas'].std().round(5)
			n      = len(grouped_results[a_model][old_id]['ious'])
			se_iou = grouped_results[a_model][old_id]['ious'].std(ddof=1) / np.sqrt(n)
			ci95   = tuple(np.round(stats.t.interval(0.95,df=4,loc=mean_iou,scale=se_iou),5))
			s = f"stage 1 id: {old_id} | iou: {mean_iou}+-{stdd_iou} | ema: {mean_ema}+-{stdd_ema}"
			s += f" | CI95: {ci95}"
			print(s)

		# PAIRWISE T-TEST (UNEQUAL VARIANCE) TO BE ABSOLUTELY SURE
		old_ids        = grouped_ids[a_model]
		n_comparisons  = len(list(itertools.combinations(old_ids,2)))
		adjusted_alpha = 0.10 / n_comparisons #Bonferroni correction

		print("\nPairwise mean t-test:")
		for id_a, id_b in itertools.combinations(old_ids,2):
			ious_a = grouped_results[a_model][id_a]['ious']
			ious_b = grouped_results[a_model][id_b]['ious']
			mean_a = ious_a.mean().round(5)
			mean_b = ious_b.mean().round(5)
			t_stat,p_val = stats.ttest_ind(ious_a,ious_b,equal_var=False)

			significant = "*" if p_val < adjusted_alpha else ""

			s2 = f"{id_a} v {id_b} | mean diff: {mean_a} - {mean_b} | t={t_stat:.5f} | p={p_val:.5f} {significant}"
			print(s2)


def get_best_stage_3(log_dir):
	'''
	Get the best cosine scheduler parameters
	'''
	with open('./hparams/stage_3.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}

	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	# GROUP RUN/EXPERIMENT ID's BY MODEL TYPE
	grouped_ids = {k:[] for k in models}
	for row in hp_list:
		grouped_ids[row['model']].append(row['id'])

	# LOAD AND GROUP DICT OF RESULTS FOR EACH MODEL RUN
	grouped_results = {k:[] for k in grouped_ids}
	for model,experiments in grouped_ids.items():
		model_results = []
		for e in experiments:
			log_file = f"{log_dir}/stage_3/epochs_{e:03}.tsv"
			result   = get_model_best_epoch(log_file)
			model_results.append(result)

		# SORT MODEL RESULTS
		ious = [r['iou'] for r in model_results]
		# emas = [r['ema'] for r in model_results]
		sorted_ious = sorted(enumerate(ious),key=lambda x: x[1],reverse=True)
		sorted_idxs = [_[0] for _ in sorted_ious]

		print(f"\n{model}")
		print("-"*40)
		for i in sorted_idxs:
			a_result    = model_results[i]
			hparameters = indexed_hp_list[int(a_result['id'])]
			s = f"id: {a_result['id']:03} | "
			s += f"iou: {a_result['iou'][0]:.5f} ep {a_result['iou'][1]:02} | "
			s += f"ema: {a_result['ema'][0]:.5f} ep {a_result['ema'][1]:02} | "
			s += f"eta_min: {hparameters['eta_min']} | cycles: {hparameters['cycles']} | "
			s += f"wdecay: {hparameters['decay']}"
			print(s)

		# plot_training_log() for best model.
		best_idx = sorted_idxs[0]
		best_id  = model_results[best_idx]['id']
		plot_training_log(f"{log_dir}/stage_3/epochs_{best_id:03}.tsv")


def get_base_test_results(log_dir):
	'''
	Get test set results for best 4 models found after stage 3.
	This is the main result.
	'''
	with open('./hparams/stage_3.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}

	best_model_indices = [0,1,2,3] # MISSING!!

	for i in best_model_indices:
		hparams = indexed_hp_list[i]

		# LOAD TEST LOG
		log_file = f"{log_dir}/stage_3/test_{i:03}.tsv"
		test_iou,test_precision,test_recall = read_test_file(log_file)

		# PRINT
		print(f"id: {i} | {hparams['model']} | ",end='')
		print(f"iou: {test_iou} | prec: {test_precision} | recall: {test_recall} ")


def get_best_stage_4(log_dir):
	'''
	Parameter size ablation.
	Run through all 16 parameter size combinations of the original 4 models.
	'''
	# --------------------------------------------------
	# LOAD & SET STRINGS
	# --------------------------------------------------
	with open('./hparams/stage_4.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}
	
	models = ["UNet_CNN_CNN","UNet_ViT_CNN","UNet_CNN_ViT","UNet_ViT_ViT"]

	# 4 EXPERIMENTS/RUN PER MODEL KIND
	grouped_ids = {k:[] for k in models}
	for row in hp_list:
		grouped_ids[row['model']].append(row['id'])

	# --------------------------------------------------
	# VALIDATION RESULTS
	# --------------------------------------------------
	# GROUP BY MODEL TO ENSURE SORTING
	grouped_results = {k:[] for k in models}
	for model,experiments in grouped_ids.items():
		for e in experiments:
			log_file = f"{log_dir}/stage_4/epochs_{experiment:03}.tsv" # <--- fails if no log
			result   = get_model_best_epoch(log_file)
			grouped_results[model].append(result)

	# PRINT GROUPED RESULTS
	for m in grouped_results:

		# PRINT GROUPED
		print(f"\n{m}")
		print("-"*40)

		for score_dict in grouped_results[m]:

			# get score hyperparameters
			hparams = indexed_hp_list[int(score_dict['id'])]
			vit_n = hparams['vit_layers']
			cnn_n = hparams['cnn_layers']
			chans = hparams['channels']
			if vit_n == 1:
				if chans == 32:
					size = "Base"
				else:
					size = "Tiny"
			else:
				if chans == 32:
					size = "Large"
				else:
					size = "Small"

			# print
			s = f"id: {score_dict['id']} | iou: {score_dict['iou']} | "
			s += f"ema: {score_dict['ema']} | "
			# s + f"{vit_n} | {cnn_n} | {chans}"
			s += f"{size}"
			print(s)


	# --------------------------------------------------
	# TEST RESULTS
	# --------------------------------------------------
	for model, experiments in group_ids.items():

		# PRINT GROUPED
		print(f"\n{model}")
		print("-"*40)

		for e in experiments:

			# LOAD TEST LOG
			log_file = f"{log_dir}/stage_4/test_{e:03}.tsv"
			test_iou,test_ppv,test_tpr = read_test_file(log_file)

			# GET CORRESPONDING HYPERPARAMETER DICT
			hparams = indexed_hp_list[e]
			vit_n = hparams['vit_layers']
			cnn_n = hparams['cnn_layers']
			chans = hparams['channels']
			if vit_n == 1:
				if chans == 32:
					size = "Base "
				else:
					size = "Tiny "
			else:
				if chans == 32:
					size = "Large"
				else:
					size = "Small"

			#PRINT
			s = f"id: {e} | {size} | {test_iou} | {test_precision} | {test_recall} "
			print(s)


def get_best_stage_5():
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

	# check_log_dir(log_dir)
	# get_best_stage_1(log_dir)
	# get_best_stage_2(log_dir)
	get_best_stage_3(log_dir)