'''
Train (arms B, C) or fine-tune (arm D) a UNet on the Sen1Floods11 .npy caches
built by dataloader.build_cached_s1f11(). A copy of train.py, with:
	- training data: the hand-labeled 'train' cache (all 252 chips) or the weakly
	  labeled 'weak' cache (no Bolivia)
	- validation and model selection on the hand-labeled 'valid' cache
	- pixels masked 127 in the caches become -1 (S1F11_IGNORE_INDEX) in the GPU
	  transforms; the loss and the confusion matrix leave them out
	- tiles with no labeled pixel are dropped (an all-ignored batch gives a NaN loss)
	- optional initial weights (arm D: an S2DW checkpoint from train.py)

Hyperparameter rows (JSON lines, one run per line) take train.py's fields plus:
	train_set        'hand' | 'weak'       cache to train on (s1f11_train / s1f11_weak)
	mean_stdd        's2dw' | 'hand' | 'weak'  input normalization (dataloader.S1F11_MEAN_STDD).
	                 From scratch it must equal train_set; with init_checkpoint it must be
	                 the normalization the initial model was trained with (s2dw for train.py
	                 checkpoints). eval_sen1floods11.py reads the same field.
	init_checkpoint  (optional) .pth.tar whose weights initialize the model (same
	                 architecture). --init-checkpoint overrides it.
	warmup_epochs    (optional, default 10 as in train.py) linear LR warmup length.
	                 experiments.py sets hand 1500 epochs (150 warmup), weak 100 (10).
The number of hand-labeled chips (k) is not a JSON hyperparameter: rows with 'k' or
'subset_seed' are rejected. Training on k-chip subsets is left to a separate
train-and-validate loop (several k per run); select_tiles() already draws them.
Example rows:
	B: {"id": 100, ..., "train_set": "hand", "mean_stdd": "hand", "seed": 1}
	C: {"id": 200, ..., "train_set": "weak", "mean_stdd": "weak", "seed": 1}
	D: {"id": 300, ..., "train_set": "hand", "mean_stdd": "s2dw", "seed": 1,
	    "init_checkpoint": "/model_pvc/experiments_1/best_007_e42.pth.tar"}

Usage (from models/; --data-dir holds the s1f11_* caches):
	python3 train_s1f11.py --data-dir /cache --net-dir ./nets --log-dir ./logs \\
		-p hparams/<file>.json --id 100 [--device 0] [--num-workers 2] [--init-checkpoint PATH]
Writes best_{id:03}_e{epoch:02}.pth.tar (best water IoU on hand 'valid', after epoch 5),
{log-dir}/epochs_{id:03}.tsv (same columns as train.py) and {log-dir}/chips_{id:03}.txt
(the training chips).
'''
import os
import numpy as np
import torch
import random
import time
import argparse
import json
from functools import wraps
import math

import models
import dataloader

####################################################################################################
# SET GLOBAL VARS FROM ENV OR ARGS
####################################################################################################
LOG_DIR   = None
MODEL_DIR = None

IGNORE      = dataloader.S1F11_IGNORE_INDEX # -1: CLASS INDEX OF UNLABELED PIXELS AFTER THE GPU TRANSFORMS
TRAIN_CACHE = {'hand': 's1f11_train', 'weak': 's1f11_weak'}
VALID_CACHE = 's1f11_valid'


####################################################################################################
# CLASSES
####################################################################################################
class Logger():
	def __init__(self,path,n_classes):
		'''
		path: str
			The file path to the text file where we log.

		head: [str]
			The column names to be included.
		'''
		self.path = path
		self.n_classes = n_classes

		header = ['tloss','vloss']
		per_class = ('tacc','ttpr','tppv','tiou','vacc','vtpr','vppv','viou')
		for prefix in per_class:
			header += [f'{prefix}{c}' for c in range(n_classes)]

		self.header = header
		self.per_class = per_class

		with open(self.path,'w') as fp:
			fp.write('\t'.join(header)+'\n')


	def log(self,metrics):
		'''
		metrics: Dict
		dict {'tloss':..., 'vloss':...,'tacc':tr_acc, 'tiou':tr_iou, ...}
		'''
		row = [f"{metrics['tloss']:.5f}",f"{metrics['vloss']:.5f}"]
		for prefix in self.per_class:
			row += [f'{metrics[prefix][c]:.5f}' for c in range(self.n_classes)]

		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


class RecentBestTracker:
	'''
	Keeps track of 'n' .pth checkpoints saved as best.
	Updates queue and removes files no longer needed.
	'''

	def __init__(self,n=3):
		self.n = n
		self.paths = [] #FIFO queue for best 3 recent

	def update(self,path):
		self.paths.append(path)
		if len(self.paths) > self.n:
			old_path = self.paths.pop(0)
			if os.path.exists(old_path):
				os.remove(old_path)

	def epochs(self):
		return ", ".join([p.split('_')[-1][1:3] for p in self.paths])


####################################################################################################
# SOME HELPER FUNCTIONS
####################################################################################################
def parse_args():
	'''
	Load args passed for hyperparameter file, cache dir, log dir, model
	checkpoint dir and optional initial weights.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,
		help='Directory with the Sen1Floods11 caches (s1f11_{train,weak,valid}_{bands,masks}.npy + _ids.txt).')
	required.add_argument('--net-dir',required=True,
		help='Output directory for trained model weights.')
	required.add_argument('--log-dir',required=True,
		help='Training logs.')
	required.add_argument('-p','--params',required=True,
		help='Path to JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,
		help='Model id number in JSON hyperparameter file.')

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--device',required=False,type=int,default=0,
		help='GPU id to train in, if different than 0. Useful to select a gpu in a multi-gpu machine.')
	optional.add_argument('--num-workers',required=False,type=int,default=2,
		help='Sets num_workers for training and validation dataloaders.')
	optional.add_argument('--init-checkpoint',required=False,default=None,
		help="Initial weights (.pth.tar). Overrides the row's init_checkpoint.")

	# LOAD
	args = parser.parse_args()

	# CHECK HERE
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"
	assert args.device >= 0, f"Got negative arg for GPU id {args.device}"
	if args.device > 0:
		assert args.device < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."

	# SET GLOBAL VARIABLES
	global LOG_DIR
	global MODEL_DIR
	LOG_DIR   = args.log_dir
	MODEL_DIR = args.net_dir
	return args


def save_checkpoint(path,model,optim,epoch,t_loss,v_loss,HP,best=False):
	'''
	Saves model+optim state as .pth.tar, plus the normalization and training
	data of the run so a later fine-tune or evaluation can check them.
	'''
	if best == True:
		save_path = f'{path}/best_{model.model_id:03}_e{epoch:02}.pth.tar'
	else:
		save_path = f'{path}/chkp_{model.model_id:03}_e{epoch:02}.pth.tar'

	# SAVE UNCOMPILED IF ALREAD COMPILED
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

	# SET CHECKPOINT AND WRITE
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'mean_stdd': HP['mean_stdd'],
					'train_set': HP['train_set'],
					'init_checkpoint': HP['init_checkpoint'],
					'model_state_dict': raw_model.state_dict(),
					'optim_state_dict': optim.state_dict()}
	torch.save(checkpoint,save_path)

	# RETURN PATH STR
	return save_path


def load_initial_weights(net,path,mean_stdd):
	'''
	Loads a train.py / train_s1f11.py checkpoint into net (same architecture,
	strict). train.py checkpoints carry no 'mean_stdd' and are S2DW-normalized.
	'''
	assert os.path.isfile(path), f"Initial checkpoint not found: {path}"
	ckpt  = torch.load(path,map_location='cpu')
	state = {k.removeprefix('_orig_mod.'): v for k,v in ckpt['model_state_dict'].items()}
	net.load_state_dict(state)
	ckpt_norm = ckpt.get('mean_stdd','s2dw')
	assert ckpt_norm == mean_stdd, \
		f"{path} was trained with mean_stdd '{ckpt_norm}' but this run normalizes with '{mean_stdd}'"
	print(f"Initial weights: {path} (epoch {ckpt.get('epoch','?')}, normalization {ckpt_norm})")


def set_seed(seed,cuda=True):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)  # If using CUDA
		torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False #Am I losing speed here?
	os.environ['PYTHONHASHSEED'] = str(seed)


@torch.no_grad()
def calculate_metrics(confmat):
	'''
	Calculate precision, recall, accuracy, and IoU for a confusion matrix tensor.
	'''
	TP = confmat.diagonal()
	FP = confmat.sum(dim=0) - TP #this is silly but explicit
	FN = confmat.sum(dim=1) - TP
	TN = confmat.sum() - TP - FP - FN

	# the metrics
	ppv = TP / (TP + FP).clamp(min=1) #precision
	tpr = TP / (TP + FN).clamp(min=1) #recall
	acc = (TP+TN) / (TP+FN+FP+TN).clamp(min=1) #accuracy
	iou = TP / (TP + FN + FP).clamp(min=1) #Intersection-over-Union

	return ppv,tpr,acc,iou


@torch.no_grad()
def update_confusion_matrix(confmat,T,Y,n_classes):
	'''
	Update a confusion matrix tensor in gpu. Per-pixel classification.
	Pixels with T == IGNORE go to an extra bin that is dropped, so they are
	not counted (no boolean indexing, which would sync with the CPU).
	'''
	# [0,1,2,3] = [TN,FP,FN,TP], n_classes**2 = IGNORED
	nn  = n_classes*n_classes
	idx = torch.where(T == IGNORE,nn,T*n_classes + Y).flatten().to(torch.int64)
	binc = torch.bincount(idx,minlength=nn+1)[:nn]
	confmat += binc.view(n_classes,n_classes)


def total_time_decorator(orig_func):
	@wraps(orig_func)
	def wrapper(*args, **kwargs):
		total_time_start = time.perf_counter()
		orig_func(*args,**kwargs)
		total_time = time.perf_counter() - total_time_start
		print(f'TOTAL TRAINING TIME: {total_time:.2f}s')

	return wrapper


def load_hyperparameters(args):

	# LOAD FILE
	with open(args.params,'r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line.strip()]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	HP = {**hp_list_indexed[args.id]}

	# OPTIONAL FIELDS
	HP['init_checkpoint'] = args.init_checkpoint or HP.get('init_checkpoint')
	HP['warmup_epochs']   = HP.get('warmup_epochs',10)

	# CHECK DICT
	try:
		# CHECK ENCODER/DECODER STRINGS MATCH models.py
		assert HP['encoder'] in models._ENCODERS, f"Incorrect encoder {HP['encoder']} in hyperparameters"
		assert HP['decoder'] in models._DECODERS, f"Incorrect decoder {HP['decoder']} in hyperparameters"

		# CHECK MODEL SIZE PARAMS -- vit_layers=0 ONLY FOR ALL-CNN MODELS
		all_cnn = HP['encoder'] == HP['decoder'] == 'cnn'
		assert HP['cnn_layers'] in [2,3], f"Incorrect # of conv layers {HP['cnn_layers']} in hyperparameters."
		assert HP['vit_layers'] in ([0] if all_cnn else [1,2]), f"Incorrect # of ViT layers {HP['vit_layers']} in hyperparameters."
		assert HP['init_chan'] in [16,32,64], f"Incorrect # of channels {HP['init_chan']} in hyperparameters."

		# SEN1FLOODS11 FIELDS
		assert HP.get('train_set') in TRAIN_CACHE, f"train_set must be one of {list(TRAIN_CACHE)}, got {HP.get('train_set')}"
		assert HP.get('mean_stdd') in dataloader.S1F11_MEAN_STDD, \
			f"mean_stdd must be one of {list(dataloader.S1F11_MEAN_STDD)}, got {HP.get('mean_stdd')}"
		if HP['init_checkpoint'] is None: # FROM SCRATCH: NORMALIZE WITH THE TRAINING SET'S OWN STATS
			assert HP['mean_stdd'] == HP['train_set'], \
				f"From scratch, mean_stdd ({HP['mean_stdd']}) must match train_set ({HP['train_set']})"
		assert 0 < HP['warmup_epochs'] < HP['epochs'], \
			f"warmup_epochs ({HP['warmup_epochs']}) must be in (0, epochs={HP['epochs']})"
		for key in ('k','subset_seed'): # CHIP SUBSETS ARE NOT SET PER ROW (SEE MODULE DOCSTRING)
			assert key not in HP, f"'{key}' is not a JSON hyperparameter for train_s1f11.py; remove it from the row"

	except AssertionError:
		print(f"hparams file:  {args.params}")
		print(f"model id/line: {HP['id']}")
		print("-"*60)
		raise

	return HP


def select_tiles(data_dir,cache,masks,k=None,subset_seed=None):
	'''
	Indices of the tiles to use. Tiles with no labeled pixel are dropped.
	k=None keeps every chip. With k (for a k-subset training loop), keeps the
	tiles of k chips: the chips that have any labeled pixel, sorted, permuted
	with subset_seed, first k -- nested across k for one subset_seed, and the
	same chips for any run with the same (k, subset_seed).
	Returns (tile indices, chip ids used).
	'''
	with open(f"{data_dir}/{cache}_ids.txt",'r') as fp:
		ids = fp.read().split('\n')
	assert len(ids) == len(masks), f"{cache}: {len(ids)} ids for {len(masks)} tiles"
	chip_of = np.array([i.rsplit('_',2)[0] for i in ids])

	# TILES WITH AT LEAST ONE LABELED PIXEL
	has_label = np.array([(m != dataloader.S1F11_IGNORE).any() for m in masks])

	# CHIP SUBSET
	chips = sorted(set(chip_of[has_label]))
	if k is not None:
		assert k <= len(chips), f"k={k} but only {len(chips)} chips have labeled pixels"
		rng   = np.random.default_rng(subset_seed)
		chips = sorted(np.array(chips)[rng.permutation(len(chips))[:k]].tolist())

	keep = np.flatnonzero(has_label & np.isin(chip_of,chips))
	print(f"[{cache}] {len(ids)} tiles, {int((~has_label).sum())} without labeled pixels | "
		f"using {len(keep)} tiles from {len(chips)} chips" + (f" (k={k}, subset_seed={subset_seed})" if k else ""))
	return keep,chips


def format_stdout_metrics(prefix, loss, acc, iou, n_classes):
	s = f'[{prefix}] LOSS: {loss:.5f} | ACC_{len(acc)-1}: {acc[-1]:.5f}'
	if n_classes > 2:
		s += f' | mIoU: {iou.mean().item():.5f}'
	else:
		s += f' | IoU_0: {iou[0]:.5f} | IoU_1: {iou[1]:.5f}'
	return s


def set_scheduler(epochs,optimizer,warmup_steps=10):
	# SCHEDULER -- LINEAR WARMUP + COSINE ANNEALING W/O RESTARTS
	# AS A FACTOR ON EACH GROUP'S OWN LRATE (SAME FORM AS THE OTHER TRAINING SCRIPTS)
	cosine_steps = epochs-warmup_steps

	# ADJUST OP
	min_factor = 1.0 / cosine_steps
	cosine_factor = lambda e: min_factor + (1.0-min_factor) * 0.5 * (1.0 + math.cos(math.pi * min(e,cosine_steps) / cosine_steps))

	# COMBINE CLASSES
	warmup_sched = torch.optim.lr_scheduler.LinearLR(
		optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.LambdaLR(
		optimizer,lr_lambda=cosine_factor)
	scheduler = torch.optim.lr_scheduler.SequentialLR(
		optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	return scheduler


####################################################################################################
# TRAININING+VALIDATION
####################################################################################################
def train(model,dataloader,aug,optimizer,loss_fn,device,n_classes=2):
	'''
	One training epoch. Returns a dict with the epoch loss and per-class metric tensors:
	{'loss': float, 'acc': tensor, 'tpr': tensor, 'ppv': tensor, 'iou': tensor}
	The loss is averaged over labeled pixels.
	'''
	# Confusion matrix in GPU to avoid sync
	gpu_mat = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

	# TRAIN LOGS
	loss_sum  = torch.zeros(1,device=device)
	pixel_sum = torch.zeros(1,device=device)

	#LOOP
	model.train()
	for X,T in dataloader:

		#TO DEVICE
		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)

		X,T = aug(X,T) # 127 -> IGNORE (-1)

		# FORWARD
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
			output = model(X)
			loss   = loss_fn(output,T) # MEAN OVER LABELED PIXELS

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# METRICS -- Loss, weighted by labeled pixels
		n_px       = (T != IGNORE).sum()
		loss_sum  += loss.detach() * n_px
		pixel_sum += n_px

		# METRICS -- Confusion matrix in gpu
		Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
		update_confusion_matrix(gpu_mat,T,Y,n_classes)

	# EPOCH METRICS
	loss_epoch  = (loss_sum/pixel_sum.clamp(min=1)).item() #------------cpu-gpu sync
	ppv,tpr,acc,iou = calculate_metrics(gpu_mat.cpu()) #---------------cpu-gpu sync
	return {'loss': loss_epoch, 'acc': acc, 'tpr': tpr, 'ppv': ppv, 'iou': iou}


@torch.no_grad()
def validate(model,dataloader,aug,loss_fn,device,n_classes=2):
	'''
	One validation pass. Returns a dict with the same keys as train().
	'''
	# Confusion matrix in GPU to avoid sync
	gpu_mat = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

	# VALIDATION LOGS
	loss_sum  = torch.zeros(1,device=device)
	pixel_sum = torch.zeros(1,device=device)

	# LOOP
	model.eval()
	for X,T in dataloader:

		# TO DEV
		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)

		X,T = aug(X,T) # 127 -> IGNORE (-1)

		# FORWARD
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
			output = model(X)
			loss   = loss_fn(output,T)
		Y = output.argmax(axis=1) #hard-prediction

		# METRICS -- Loss, weighted by labeled pixels
		n_px       = (T != IGNORE).sum()
		loss_sum  += loss * n_px
		pixel_sum += n_px

		# METRICS -- Confusion matrix
		update_confusion_matrix(gpu_mat,T,Y,n_classes)

	# EPOCH METRICS
	loss_epoch  = (loss_sum/pixel_sum.clamp(min=1)).item() #------------cpu-gpu sync
	ppv,tpr,acc,iou = calculate_metrics(gpu_mat.cpu()) #---------------cpu-gpu sync
	return {'loss': loss_epoch, 'acc': acc, 'tpr': tpr, 'ppv': ppv, 'iou': iou}


@total_time_decorator
def train_and_validate(model,dataloaders,tr_aug,va_aug,optimizer,loss_fn,scheduler,device,HP,epochs=100,n_classes=2):

	# TRAINING/VALIDATION LOGGING
	log_file_path = f'{LOG_DIR}/epochs_{model.model_id:03}.tsv'
	logger        = Logger(log_file_path,n_classes)

	# BEST MODEL/EPOCH METRICS
	best_epoch = 0
	best_iou   = 0.0
	recent_best_iou = RecentBestTracker(n=3)

	for epoch in range(epochs):

		# STDOUT
		print(f'\nEpoch {epoch}/{epochs-1}')
		print('-'*80,flush=True)

		# EPOCH TIME
		epoch_start_time = time.perf_counter()

		# TRAINING
		tr_result = train(model,dataloaders['training'],tr_aug,optimizer,loss_fn,device,n_classes)
		scheduler.step()
		print(format_stdout_metrics('T',tr_result['loss'],tr_result['acc'],tr_result['iou'],n_classes))

		# VALIDATION
		va_result = validate(model,dataloaders['validation'],va_aug,loss_fn,device,n_classes)
		print(format_stdout_metrics('V',va_result['loss'],va_result['acc'],va_result['iou'],n_classes))

		############################################################
		# LOG EPOCH
		############################################################
		# EPOCH TIME
		epoch_time = time.perf_counter() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')

		# LOG THIS EPOCH RESULTS -- t/v PREFIX ON EACH RESULT KEY (tloss, viou, ...)
		logger.log({**{f't{k}': v for k,v in tr_result.items()},
					**{f'v{k}': v for k,v in va_result.items()}})

		# EPOCH VALIDATION IoU -- WATER IoU ON HAND 'valid'
		epoch_iou = va_result['iou'][1].item()

		# MAX RAW IoU
		if epoch >= 5 and best_iou < epoch_iou:
			best_iou   = epoch_iou
			best_epoch = epoch
			chkpt_path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,tr_result['loss'],va_result['loss'],HP,best=True)
			recent_best_iou.update(chkpt_path)

	############################################################
	# LOG OVERALL
	############################################################
	print(f'\nBest validation IoU:    {best_iou:.5f} -- Epoch {best_epoch}')
	print(f'Epochs saved: {recent_best_iou.epochs()} (by iou)')
	if device.type == 'cuda':
		mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
		print(f"Peak GPU memory allocated: {mem_gb:.2f} GB")


####################################################################################################
# MAIN
####################################################################################################
if __name__ == "__main__":
	#----------- ARGV -------------
	args = parse_args()

	#---------- LOAD AND PARSE HYPERPARAMETER DICT ------------------------------------------------
	HP = load_hyperparameters(args)
	print(f"Run id {HP['id']}: train_set {HP['train_set']} | epochs {HP['epochs']} (warmup {HP['warmup_epochs']}) | seed {HP['seed']} | "
		f"normalization {HP['mean_stdd']} | init {HP['init_checkpoint'] or 'scratch'}")

	#---------- SET SEEDS -------------------------------------------------------------------------
	if HP['seed'] != 0: #If zero in dict, seed not assigned.
		set_seed(HP['seed'])

	#---------- DEVICE/GPU ------------------------------------------------------------------------
	# CPU FALLBACK ONLY FOR SMOKE TESTS (NO bfloat16 AUTOCAST, NO COMPILE)
	device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

	#---------- LOAD MODEL ------------------------------------------------------------------------
	net = models.UNet(
		HP['id'],
		encoder=HP['encoder'],
		decoder=HP['decoder'],
		in_channels=4,
		out_labels=2,
		cnn_layers=HP['cnn_layers'],
		vit_layers=HP['vit_layers'],
		init_channels=HP['init_chan']
	)
	if HP['init_checkpoint']:
		load_initial_weights(net,HP['init_checkpoint'],HP['mean_stdd'])
	net = net.to(device)

	#---------- LOSS ------------------------------------------------------------------------------
	loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)

	#---------- OPTIMIZER -------------------------------------------------------------------------
	optimizer = torch.optim.AdamW(net.parameters(),lr=HP['lrate'],weight_decay=HP["decay"])

	#---------- LEARNING RATE SCHEDULER ------------------------------------------------------------
	scheduler = set_scheduler(HP['epochs'],optimizer,warmup_steps=HP['warmup_epochs'])

	#---------- DATALOADERS ------------------------------------------------------------------------
	# Transforms (in gpu) -- 127 -> IGNORE, NORMALIZED WITH THE RUN'S mean_stdd
	tr_aug = dataloader.S1F11GPUTransform(gain=0.10,mean_stdd=HP['mean_stdd']).to(device)
	va_aug = dataloader.S1F11GPUValidationTransform(mean_stdd=HP['mean_stdd']).to(device)

	# Load a contiguous dataset to RAM -- np.ndarray
	tr_cache = TRAIN_CACHE[HP['train_set']]
	tr_bands_np,tr_masks_np = dataloader.load_cached_split(args.data_dir,split=tr_cache)
	va_bands_np,va_masks_np = dataloader.load_cached_split(args.data_dir,split=VALID_CACHE)

	# Tiles to use: all chips, no all-ignored tiles. Subset VIEWS THE ARRAYS, NO COPY
	tr_keep,tr_chips = select_tiles(args.data_dir,tr_cache,tr_masks_np)
	va_keep,_        = select_tiles(args.data_dir,VALID_CACHE,va_masks_np)
	with open(f"{LOG_DIR}/chips_{HP['id']:03}.txt",'w') as fp:
		fp.write('\n'.join(tr_chips))

	tr_dataset = torch.utils.data.Subset(dataloader.S1F11FastDataset(tr_bands_np,tr_masks_np),tr_keep.tolist())
	va_dataset = torch.utils.data.Subset(dataloader.S1F11FastDataset(va_bands_np,va_masks_np),va_keep.tolist())
	assert len(tr_dataset) >= HP['batch'], \
		f"{len(tr_dataset)} training tiles < batch {HP['batch']}: drop_last would leave no batches"

	tr_dataloader = torch.utils.data.DataLoader(
		tr_dataset,
		batch_size=HP['batch'],
		drop_last=True,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6 if args.num_workers > 0 else None,
		persistent_workers=args.num_workers > 0
	)

	va_dataloader = torch.utils.data.DataLoader(
		va_dataset,
		batch_size=HP['batch'],
		drop_last=False,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6 if args.num_workers > 0 else None,
		persistent_workers=args.num_workers > 0
	)

	dataloaders = {
		'training': tr_dataloader,
		'validation': va_dataloader
	}

	#---------- TRAINING/RUN -----------------------------------------------------------------------
	if device.type == 'cuda':
		net = torch.compile(net)

	train_and_validate(
		net,
		dataloaders,
		tr_aug,
		va_aug,
		optimizer,
		loss_fn,
		scheduler,
		device,
		HP,
		HP['epochs'],
		n_classes=2
	)
