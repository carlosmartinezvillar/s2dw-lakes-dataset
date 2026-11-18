import os
import numpy as np
import torch
import random
import time
# from tqdm import tqdm
import argparse
import json
from functools import wraps
import math

import models
import dataloader

####################################################################################################
# SET GLOBAL VARS FROM ENV OR ARGS
####################################################################################################
# __spec__ = None # DEBUG with tqdm -- temp.
DATA_DIR  = None
LOG_DIR   = None
MODEL_DIR = None
CUDA_DEV  = None
N_WORKERS = None


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
		# line = '\t'.join([f'{_:.5f}' for _ in stats])

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
	Load args passed for hyperparameter file, chip dir, log dir, model type,
	model checkpoint dir.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,
		help='Input dataset directory.')
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
	# global DATA_DIR
	global LOG_DIR
	global MODEL_DIR
	# global CUDA_DEV
	# global N_WORKERS
	# DATA_DIR  = args.data_dir
	LOG_DIR   = args.log_dir
	MODEL_DIR = args.net_dir
	# CUDA_DEV  = torch.device(f"cuda:{args.gpu}")
	# N_WORKERS = args.num_workers
	return args


def save_checkpoint(path,model,optim,epoch,t_loss,v_loss,best=False):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	# Standard path is:
	# save_path = f'{MODEL_DIR}/_{epoch:03d}.pt'
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
					'model_state_dict': raw_model.state_dict(),
					'optim_state_dict': optim.state_dict()}
	torch.save(checkpoint,save_path)

	# RETURN PATH STR
	return save_path


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

	# Add stuff
	TP = confmat.diagonal()
	FP = confmat.sum(dim=0) - TP #this is silly but explicit
	FN = confmat.sum(dim=1) - TP
	TN = confmat.sum() - TP - FP - FN
	# eps = 0.0000000001

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
	'''
	# confmat[0,0] += ((T==0) & (Y==0)).sum() #TN
	# confmat[0,1] += ((T==0) & (Y==1)).sum() #FP
	# confmat[1,0] += ((T==1) & (Y==0)).sum() #FN
	# confmat[1,1] += ((T==1) & (Y==1)).sum() #TP

	# 2x2
	# for k in range(n_classes*n_classes):
		# i = k // n_classes #row
		# j = k % n_classes #col
		# confmat[i,j] += ((T==i) & (Y==j)).sum()

	# Vectorized
	# [0,1,2,3] = [TN,FP,FN,TP]
	idx = (T.flatten()*n_classes + Y.flatten()).to(torch.int64)
	binc = torch.bincount(idx,minlength=n_classes*n_classes)
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
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	HP = {**hp_list_indexed[args.id]}

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

	except AssertionError:
		print(f"hparams file:  {args.params}")
		print(f"model id/line: {HP['id']}")
		print("-"*60)
		raise

	return HP


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

	# ADJUSTMENT STEPS
	# warmup_steps = 10
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
	'''
	# Confusion matrix in GPU to avoid sync
	gpu_mat = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

	# TRAIN LOGS
	loss_sum   = torch.zeros(1,device=device)
	sample_sum = torch.zeros(1,device=device)

	#LOOP
	model.train()
	for X,T in dataloader:

		#TO DEVICE
		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)

		X,T = aug(X,T)

		# FORWARD
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# METRICS -- Loss
		loss_sum   += loss.detach() * X.size(0)
		sample_sum += X.size(0)

		# METRICS -- Confusion matrix in gpu
		Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
		update_confusion_matrix(gpu_mat,T,Y,n_classes)

	# EPOCH METRICS
	loss_epoch  = (loss_sum/sample_sum).item() #------------------------cpu-gpu sync
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
	loss_sum   = torch.zeros(1,device=device)
	sample_sum = torch.zeros(1,device=device)

	# LOOP
	model.eval()
	for X,T in dataloader:

		# TO DEV
		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)

		X,T = aug(X,T)

		# FORWARD
		with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T)
		Y = output.argmax(axis=1) #hard-prediction

		# METRICS -- Loss
		loss_sum   += loss * X.size(0)
		sample_sum += X.size(0)

		# METRICS -- Confusion matrix
		update_confusion_matrix(gpu_mat,T,Y,n_classes)

	# EPOCH METRICS
	loss_epoch  = (loss_sum/sample_sum).item() #------------------------cpu-gpu sync
	ppv,tpr,acc,iou = calculate_metrics(gpu_mat.cpu()) #---------------cpu-gpu sync
	return {'loss': loss_epoch, 'acc': acc, 'tpr': tpr, 'ppv': ppv, 'iou': iou}


@total_time_decorator
def train_and_validate(model,dataloaders,tr_aug,va_aug,optimizer,loss_fn,scheduler,device,epochs=100,n_classes=2):

	# AUTOMATIC MIXED PRECISION -- switched to bfloat16
	# scaler = torch.amp.GradScaler("cuda",enabled=True,init_scale=1024)

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

		# EPOCH VALIDATION IoU -- IoU/mIoU
		epoch_iou = va_result['iou'][1].item() #true label iou for 2 classes

		# MAX RAW IoU
		if epoch >= 5 and best_iou < epoch_iou:
			best_iou   = epoch_iou
			best_epoch = epoch
			chkpt_path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,tr_result['loss'],va_result['loss'],best=True)
			recent_best_iou.update(chkpt_path)

	############################################################
	# LOG OVERALL
	############################################################
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'\nBest validation IoU:    {best_iou:.5f} -- Epoch {best_epoch}')
	print(f'Epochs saved: {recent_best_iou.epochs()} (by iou)') 
	print(f"Peak GPU memory allocated: {mem_gb:.2f} GB")


####################################################################################################
# MAIN
####################################################################################################
if __name__ == "__main__":
	#----------- ARGV -------------
	args = parse_args()

	#---------- LOAD AND PARSE HYPERPARAMETER DICT ------------------------------------------------
	HP = load_hyperparameters(args)

	#---------- SET SEEDS -------------------------------------------------------------------------
	if HP['seed'] != 0: #If zero in dict, seed not assigned.
		set_seed(HP['seed'])

	#---------- DEVICE/GPU ------------------------------------------------------------------------
	device = torch.device(f'cuda:{args.device}')

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
	net = net.to(device)

	#---------- LOSS ------------------------------------------------------------------------------
	loss_fn = torch.nn.CrossEntropyLoss()

	#---------- OPTIMIZER -------------------------------------------------------------------------
	optimizer = torch.optim.AdamW(net.parameters(),lr=HP['lrate'],weight_decay=HP["decay"])

	#---------- LEARNING RATE SCHEDULER ------------------------------------------------------------
	scheduler = set_scheduler(HP['epochs'],optimizer,warmup_steps=10)

	#---------- DATALOADERS ------------------------------------------------------------------------
	# Transforms (in gpu)
	tr_aug = dataloader.GPUTransform(gain=0.10).to(device)
	va_aug = dataloader.GPUValidationTransform().to(device)

	# Load a contiguous dataset to RAM -- np.ndarray
	tr_bands_np,tr_masks_np = dataloader.load_cached_split(args.data_dir,split='training')
	va_bands_np,va_masks_np = dataloader.load_cached_split(args.data_dir,split='validation')

	# Load a contiguous dataset to RAM -- dataset
	tr_dataset = dataloader.SentinelFastDataset(tr_bands_np,tr_masks_np,n_bands=4)
	va_dataset = dataloader.SentinelFastDataset(va_bands_np,va_masks_np,n_bands=4)

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
		HP['epochs'],
		n_classes=2
	)
