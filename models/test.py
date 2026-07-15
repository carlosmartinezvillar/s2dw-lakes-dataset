import os
import numpy as np
import torch
import random
import time
import argparse
import json
from functools import wraps

import models
import dataloader

################################################################################
# GLOBAL VARS
################################################################################
DATA_DIR  = None
MODEL_DIR = None
LOG_DIR   = None
CUDA_DEV  = None

################################################################################
# HELPER FUNCTIONS
################################################################################
def parse_args():
	parser = argparse.ArgumentParser()

	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,
		help='Input dataset directory.')
	required.add_argument('--net-dir',required=True,
		help='Directory of trained model weights.')
	required.add_argument('--log-dir',required=True,
		help='Dest dir path for test results')
	required.add_argument('-p','--params',required=True,
		help='Path to JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,
		help='Model id number in JSON hyperparameter file.')

	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--gpu',required=False,type=int,default=0,
		help='GPU id to train in, if different than 0. Useful to select a gpu in a multi-gpu machine.')

	# LOAD
	args = parser.parse_args()

	# CHECK
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"
	if args.gpu != 0:
		assert args.gpu < torch.cuda.device_count()		

	# SET
	global DATA_DIR
	global MODEL_DIR
	global LOG_DIR
	global CUDA_DEV
	DATA_DIR  = args.data_dir
	MODEL_DIR = args.net_dir
	LOG_DIR   = args.log_dir
	CUDA_DEV  = torch.device(f"cuda:{args.gpu}")

	return args


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
	For 2 classes, for example, the order is set as:

	# confmat[0,0] += ((T==0) & (Y==0)).sum() #True-negative
	# confmat[0,1] += ((T==0) & (Y==1)).sum() #False-positive
	# confmat[1,0] += ((T==1) & (Y==0)).sum() #False-negative
	# confmat[1,1] += ((T==1) & (Y==1)).sum() #True-positive
	'''
	for k in range(n_classes*n_classes):
		i = k // n_classes #row
		j = k % n_classes #col
		confmat[i,j] += ((T==i) & (Y==j)).sum()


def total_time_decorator(orig_func):
	@wraps(orig_func)
	def wrapper(*args, **kwargs):
		total_time_start = start_time = time.perf_counter()
		orig_func(*args,**kwargs)
		total_time = time.perf_counter() - total_time_start
		print(f'TOTAL TRAINING TIME: {total_time:.2f}s')
	return wrapper


def log_results(path,metrics):

	n_classes = len(metrics['_ppv'])
	metric_str = metrics.keys()
	header = []
	for prefix in metric_str:
		header += [f"{prefix}{i}" for i in range(n_classes)]

	result = []
	for prefix in metric_str:
		result += [f"{metrics[prefix][c]:.5f}" for c in range(n_classes)]


	with open(path,'w') as fp:
		fp.write('\t'.join(header)+'\n')
		fp.write('\t'.join(result))

	# STDOUT
	print(f"Result file written to {path}")


def load_checkpoint(path, model):
    '''
    Loads model state from a .pth.tar checkpoint.
    '''

    # LOAD DICT/TENSORS TO DEV
    checkpoint = torch.load(path, map_location=CUDA_DEV)

    # LOAD WEIGHTS 
    model.load_state_dict(checkpoint['model_state_dict'])

    # LOAD OTHER
    # if scaler is not None:
		# scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    t_loss = checkpoint['t_loss']
    v_loss = checkpoint['v_loss']

    return model,epoch,t_loss,v_loss


@total_time_decorator
def test(model,dataloader,n_classes):

	# Count samples in GPU
	sample_sum = torch.zeros(1,device=CUDA_DEV)
	gpu_mat    = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64)
	log_path   = f"{LOG_DIR}/test_{model.model_id:03}.tsv"

	# PASS TEST SET
	model.eval()
	with torch.no_grad():
		for X,T in dataloader:

			# TO GPU
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
				output        = model(X)
				Y_soft,Y_hard = torch.max(output,1) 		

			# APPEND COUNTS
			sample_sum += X.size(0)
			update_confusion_matrix(gpu_mat,T,Y_hard)

	# GET METRICS OVER ENTIRE SET
	cpu_mat = gpu_mat.cpu() #syncs gpu-cpu
	ppv,tpr,acc,iou = calculate_metrics(cpu_mat)

	# LOG RESULT
	results = {'_acc':acc,'_tpr':tpr,'_ppv':ppv,'_iou':iou} #irrespective order
	log_results(log_path,results)


if __name__ == '__main__':

	# LOAD SCRIPT PARAMETERS
	parse_args()

	# LOAD HYPERPARAMETER DICT
	with open(args.params,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != "\n"]
	hp_list_indexed = {row['id']:row for row in HP_LIST}
	assert args.id in hp_list_indexed, f"MODEL ID '{args.id}' NOT IN HYPERPARAMETER FILE."
	HP = hp_list_indexed[args.id]

	n_bands         = HP['bands']
	n_classes       = HP['labels']
	batch_size      = HP['batch']
	model_class_str = HP['model']
	model_id        = HP['id']

	# LOAD MODEL TYPE & CHECKPOINT WEIGHTS
	checkpoint_path = f"{MODEL_DIR}/best_{model_id:03}.pth.tar"
	net = eval(f"models.{model_class_str}({model_id},{n_bands},{n_classes})")
	net,_,_,_ = load_checkpoint(checkpoint_path,net)
	net = net.to(CUDA_DEV)

	# DATALOADER
	dataset = dataloader.SentinelDataset(f"{DATA_DIR}/testing",n_bands,n_classes)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		drop_last=False,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
		prefetch_factor=10
	)

	# RUN TEST
	test(net,dataloader,n_classes)