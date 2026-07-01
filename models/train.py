import os
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import random
import time
# from tqdm import tqdm
import argparse
import json
from functools import wraps
import inspect

import models
import dataloader

####################################################################################################
# SET GLOBAL VARS FROM ENV OR ARGS
####################################################################################################
# __spec__ = None # DEBUG with tqdm -- temp.

parser = argparse.ArgumentParser()

required = parser.add_argument_group('Required arguments')
required.add_argument('--data-dir',required=True,
	help='Input dataset directory.')
required.add_argument('--net-dir',required=True,
	help='Output directory for trained model weights.')
required.add_argument('--log-dir',required=True,default=None,
	help='Training logs.')
required.add_argument('-p','--params',required=True,default=None,
	help='Path to JSON hyperparameter file.')
required.add_argument('--id',required=True,type=int,default=0,
	help='Model id number in JSON hyperparameter file.')

optional = parser.add_argument_group('Optional arguments')
optional.add_argument('--gpu',required=False,type=int,default=0,
	help='GPU id to train in, if different than 0. Useful to select a gpu in a multi-gpu machine.')
optional.add_argument('--full',required=False,action='store_true',default=False,
	help='Train on both training and validation sets (training final model).')


args = parser.parse_args()

DATA_DIR  = args.data_dir
LOG_DIR   = args.log_dir
MODEL_DIR = args.net_dir
CUDA_DEV  = None

####################################################################################################
# SOME HELPER FUNCTIONS
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
		per_class = ('tacc','tiou','vacc','vtpr','vppv','viou')
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


####################################################################################################
# SOME HELPER FUNCTIONS
####################################################################################################
def save_checkpoint(path,model,optim,scaler,epoch,t_loss,v_loss,best=False):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	# save_path = f'{MODEL_DIR}/state_{epoch:03d}.pt'
	if best == True:
		save_path = f'{path}/best_{model.model_id:03}.pth.tar'
	else:
		save_path = f'{path}/model_{model.model_id:03}_e{epoch:02}.pth.tar'
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'model_state_dict': model.state_dict(),
					'optim_state_dict': optim.state_dict(),
					'scaler_state_dict': scaler.state_dict()}
	torch.save(checkpoint,save_path)


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

	for k in range(n_classes*n_classes):
		i = k // n_classes #row
		j = k % n_classes #col
		confmat[i,j] += ((T==i) & (Y==j)).sum()


def total_time_decorator(orig_func):
	@wraps(orig_func)
	def wrapper(*args, **kwargs):
		total_time_start = time.time()
		orig_func(*args,**kwargs)
		total_time = time.time() - total_time_start
		print(f'TOTAL TRAINING TIME: {total_time:.2f}s')

	return wrapper


def print_exploding_layers(total_norm,model):
	if torch.isinf(total_norm) or torch.isnan(total_norm):
	    for name, param in model.named_parameters():
	        if param.grad is not None:
	            param_norm = param.grad.data.norm(2)
	            if torch.isinf(param_norm) or torch.isnan(param_norm):
	                print(f"--- Layer: {name} | Norm: {param_norm}")		


def format_stdout_metrics(prefix, loss, acc, iou, n_classes):
	s = f'[{prefix}] LOSS: {loss:.5f} | ACC: {acc[-1]:.5f}'
	if n_classes > 2:
		s += f' | mIoU: {iou.mean().item():.5f}'
	else:
		s += f' | IoU_0: {iou[0]:.5f} | IoU_1: {iou[1]:.5f}'
	return s

####################################################################################################
# TRAININING ON FULL DATASET? -- MISSING
####################################################################################################
def train_full_set(model,dataloaders,optimizer,loss_fn,scaler,scheduler,epochs=100,n_classes=2):
	'''
	Train the model with the train+validation datasets combined.
	'''
	pass


####################################################################################################
# TRAININING+VALIDATION
####################################################################################################
@total_time_decorator
def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler,epochs=50,n_classes=2):

	# AUTOMATIC MIXED PRECISION
	scaler = torch.amp.GradScaler("cuda",enabled=True,init_scale=1024)

	# LOGS
	log_file_path = f'{LOG_DIR}/epochs_{model.model_id:03}.tsv'
	logger        = Logger(log_file_path,n_classes)
	best_iou   = 0.0
	best_epoch = 0

	for epoch in range(epochs):

		# Confusion matrices in GPU to avoid sync
		gpu_mat_tr = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64) 
		gpu_mat_va = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64)

		# Single epoch time
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch}/{epochs-1}')
		print('-'*80,flush=True)

		############################################################
		# TRAINING
		############################################################
		# LOGS		
		loss_sum_tr   = torch.zeros(1,device=CUDA_DEV)
		sample_sum_tr = torch.zeros(1,device=CUDA_DEV)
		# sum_norms     = torch.zeros(1,device=CUDA_DEV) #gradient norms

		#LOOP
		model.train()		
		for X,T in dataloaders['training']:

			#TO DEVICE
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T)

			# BACKPROP
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			# sum_norms += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			# METRICS -- Loss
			loss_sum_tr   += loss.detach() * X.size(0)
			sample_sum_tr += X.size(0)

			# METRICS -- Confusion matrix in gpu
			Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
			T = T.detach()
			update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)

		#SCHEDULER UPDATE
		if scheduler is not None:
			scheduler.step()

		# TRAINING METRICS FOR LOG
		loss_tr    = (loss_sum_tr/sample_sum_tr).item() #-------------sync
		cpu_mat_tr = gpu_mat_tr.cpu() #-------------------------------sync
		# avg_norm   = sum_norms.item()/len(dataloaders['training']) #--sync
		tr_ppv,tr_tpr,tr_acc,tr_iou = calculate_metrics(cpu_mat_tr) #tensor,result per class 
		print(format_stdout_metrics('T',loss_tr,tr_acc,tr_iou,n_classes))
		# norms -------------------------------------------------- remove
		# print(f"AVG NORM: {avg_norm:.5f}")
		
		############################################################
		# VALIDATION
		############################################################
		# LOGS		
		loss_sum_va   = torch.zeros(1,device=CUDA_DEV)
		sample_sum_va = torch.zeros(1,device=CUDA_DEV)

		# LOOP
		model.eval()
		with torch.no_grad():
			for X,T in dataloaders['validation']:

				# TO DEV
				X = X.to(CUDA_DEV,non_blocking=True)
				T = T.to(CUDA_DEV,non_blocking=True)

				# FORWARD
				with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
					output = model(X)
					loss   = loss_fn(output,T)
				Y_soft,Y   = torch.max(output,1) #soft-prediction, hard-prediction

				# METRICS -- Loss
				loss_sum_va   += loss.detach() * X.size(0)
				sample_sum_va += X.size(0)

				# METRICS -- Confusion matrix
				update_confusion_matrix(gpu_mat_va,T,Y,n_classes)


		# VALIDATION METRICS FOR LOG
		loss_va    = (loss_sum_va / sample_sum_va).item() #-------------sync
		cpu_mat_va = gpu_mat_va.cpu() #---------------------------------sync
		va_ppv,va_tpr,va_acc,va_iou = calculate_metrics(cpu_mat_va)
		print(format_stdout_metrics('V',loss_va,va_acc,va_iou,n_classes))

		############################################################
		# LOG EPOCH
		############################################################
		# TIME
		epoch_time = time.time() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')

		# RESULTS
		logger.log({'tloss': loss_tr, 'vloss': loss_va,
			'tacc': tr_acc, 'tiou': tr_iou, 'vacc': va_acc,
			'vtpr': va_tpr, 'vppv': va_ppv, 'viou': va_iou})

		# SAVE MODEL
		if n_classes > 2:
			epoch_iou = va_iou.mean().item() #mIoU for 3+ classes
		else:
			epoch_iou = va_iou[1] #true label iou for 2 classes

		if best_iou < epoch_iou:
			best_iou   = epoch_iou
			best_epoch = epoch
			save_checkpoint(MODEL_DIR,model,optimizer,scaler,epoch,loss_tr,loss_va,best=True)

	print(f'Best validation IoU: {best_iou:.5f} -- Epoch {best_epoch}')



if __name__ == "__main__":

	#---------- LOAD AND PARSE HYPERPARAMETER DICT ------------------------------------------------
	assert os.path.isfile(args.params), "INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != "\n"]
	assert len(HP_LIST) > 0, "GOT EMPTY JSON FILE."

	# SEARCH BY ID
	hp_list_indexed = {row['ID']:row for row in HP_LIST}
	assert args.id in hp_list_indexed, f"MODEL ID '{args.id}' NOT IN HYPERPARAMETER FILE."
	HP = hp_list_indexed[args.id]

	#---------- GPU  ------------------------------------------------------------------------------
	# assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
	if torch.cuda.is_available():
		if args.gpu > 0:
			assert args.gpu < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."
		CUDA_DEV = torch.device(f"cuda:{args.gpu}")
	else:
		CUDA_DEV = torch.device("cpu") 

	#---------- SET ALL SEEDS ----------------------------------------------------------------------
	assert HP['SEED'] in (0,1), "INCORRECT SEED VALUE IN JSON HYPEPARAMETER DICT."
	if HP['SEED'] == True:
		set_seed(476)

	#---------- INPUT BANDS -----------------------------------------------------------------------
	assert HP['BANDS'] in [3,4],"INCORRECT BAND NR IN JSON HYPERPARAMETER DICT."
	n_bands = HP['BANDS']

	#---------- OUTPUT CHANNELS -------------------------------------------------------------------
	assert HP['OUTPUTS'] in [2,3], "INCORRECT # OF CLASSES SET IN JSON HYPERPARAMETER DICT."
	n_classes = HP['OUTPUTS']

	#---------- MODEL -----------------------------------------------------------------------------
	model_classes = [name for name,obj in inspect.getmembers(models,inspect.isclass)]
	assert HP['MODEL'] in model_classes, "INCORRECT MODEL STRING IN HYPERPARAMETER DICT"
	net = eval(f"models.{HP['MODEL']}({HP['ID']},in_channels={n_bands},out_labels={n_classes})")
	net = net.to(CUDA_DEV)
	net = torch.compile(net)

	#---------- LOSS ------------------------------------------------------------------------------
	assert HP['LOSS'] in ["ce","ew","cw"], "INCORRECT STRING FOR LOSS IN DICT."
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	if HP['LOSS'] == "ew":
		loss_fn = None
	if HP['LOSS'] == "cw": #<<< --- Needs some work...
		loss_fn = None

	#---------- OPTIMIZER -------------------------------------------------------------------------
	# assert HP["LEARNING_RATE"] in [0.0001,0.00025,0.0005,0.00075,0.001],"LR OUT OF RANGE."
	assert HP["OPTIM"] in ["adam","sgd","adamw"], "INCORRECT STRING FOR OPTIMIZER IN DICT."
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == "sgd":
		optimizer = torch.optim.SGD(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == 'adamw':
		optimizer = torch.optim.AdamW(net.parameters(),lr=HP['LEARNING_RATE'],
			weight_decay=HP["DECAY"])

	#---------- LEARNING RATE SCHEDULER ------------------------------------------------------------
	assert HP["SCHEDULER"] in ["step","exp","none"] #making it overly explicit for logs/results
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.3)
	if HP['SCHEDULER'] == "exp":
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
	if HP['SCHEDULER'] == "none":
		scheduler = None

	#---------- DATALOADERS ------------------------------------------------------------------------
	transform = v2.Compose([
		v2.RandomHorizontalFlip(p=0.5),
		v2.RandomVerticalFlip(p=0.5)
	])

	tr_dataset = dataloader.SentinelDataset(f"{DATA_DIR}/training",
		n_bands=n_bands,
		n_labels=n_classes,
		transform=transform)

	va_dataset = dataloader.SentinelDataset(f"{DATA_DIR}/validation",
		n_bands=n_bands,
		n_labels=n_classes,
		transform=None)

	dataloaders = {
		'training': torch.utils.data.DataLoader(
			tr_dataset,
			batch_size=HP['BATCH'],
			drop_last=False,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
			prefetch_factor=10),
		'validation': torch.utils.data.DataLoader(
			va_dataset,
			batch_size=HP['BATCH'],
			drop_last=False,
			shuffle=False,
			num_workers=4,
			pin_memory=True,
			prefetch_factor=10)
	}

	#---------- TRAINING --------------------------------------------------------------------------
	train_and_validate(
		net,
		dataloaders,
		optimizer,
		loss_fn,
		scheduler,
		HP['EPOCHS'],
		n_classes
	)
