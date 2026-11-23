'''
Test a train.py / train_s1f11.py checkpoint on a cached test set.

	--test-set s2dw           S2DW testing split    {data-dir}/testing_{bands,masks}.npy
	                          SentinelFastDataset + GPUValidationTransform
	--test-set s1f11_hand     Sen1Floods11 hand-labeled 'test' (10 events)   {data-dir}/s1f11_test_*
	--test-set s1f11_bolivia  Sen1Floods11 hand-labeled 'bolivia' (unseen event)   {data-dir}/s1f11_bolivia_*
	                          S1F11FastDataset + S1F11GPUValidationTransform (127 -> -1, left out of the metrics)

Input normalization comes from the hyperparameter row: train_set (absent = train.py,
S2DW) -> s2dw / hand / weak stats. A row's 'mean_stdd' takes precedence: a fine-tuned
model (arm D: train_set 'hand', S2DW-pretrained) keeps the stats of its pretraining.
Checkpoints from train_s1f11.py also store mean_stdd, which must agree.

Metrics are pooled over all labeled pixels of the test set (per-class acc, tpr, ppv, iou),
written to {log-dir}/test_{id:03}_{test-set}.tsv. Per-chip mean IoU and the water-fraction
groups are in eval_sen1floods11.py.

Usage (from models/):
	python3 test.py --data-dir /cache --checkpoint ./nets/best_015_e21.pth.tar --log-dir ./logs \\
		-p hparams/experiments_1.json --test-set s1f11_bolivia [--gpu 0] [--save-images]
'''
import os
import numpy as np
import torch
import random
import time
import argparse
import json
from functools import wraps
from PIL import Image
import sys

import models
import dataloader

################################################################################
# GLOBAL VARS
################################################################################
DATA_DIR   = None
MODEL_PATH = None
LOG_DIR    = None
CUDA_DEV   = None

IGNORE = dataloader.S1F11_IGNORE_INDEX # -1: UNLABELED PIXELS AFTER S1F11GPUValidationTransform

# TEST SET -> CACHE NAME PASSED TO dataloader.load_cached_split
TEST_CACHE = {
	's2dw':          'testing',
	's1f11_hand':    's1f11_test',
	's1f11_bolivia': 's1f11_bolivia'}

# HP['train_set'] -> NORMALIZATION (dataloader.S1F11_MEAN_STDD). train.py ROWS HAVE NO train_set
TRAIN_SET_STATS = {
	None:   's2dw',
	'hand': 'hand',
	'weak': 'weak'}

################################################################################
# HELPER FUNCTIONS
################################################################################
def parse_args():
	parser = argparse.ArgumentParser()

	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,
		help='Directory with the cached .npy test sets (testing_*, s1f11_test_*, s1f11_bolivia_*).')
	required.add_argument('--checkpoint',required=True,
		help='Trained model weights (best_{id:03}_e{epoch:02}.pth.tar).')
	required.add_argument('--log-dir',required=True,
		help='Dest dir path for test results')
	required.add_argument('-p','--params',required=True,
		help='Path to JSON hyperparameter file.')
	required.add_argument('--test-set',required=True,choices=list(TEST_CACHE),
		help='s2dw (S2DW testing), s1f11_hand (Sen1Floods11 hand-labeled test) or s1f11_bolivia.')

	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--gpu',required=False,type=int,default=0,
		help='GPU id to train in, if different than 0. Useful to select a gpu in a multi-gpu machine.')
	optional.add_argument('--num-workers',required=False,type=int,default=4,
		help='num_workers of the test dataloader.')
	optional.add_argument('--save-images',required=False,action='store_true',default=False,
		help='Save predicted images for a model')

	# LOAD
	args = parser.parse_args()

	# CHECK
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isfile(args.checkpoint), f"No path found for checkpoint dir in {args.checkpoint}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"
	if args.gpu != 0:
		assert args.gpu < torch.cuda.device_count()

	# SET
	global DATA_DIR
	global MODEL_PATH
	global LOG_DIR
	global CUDA_DEV
	DATA_DIR   = args.data_dir
	MODEL_PATH = args.checkpoint
	LOG_DIR    = args.log_dir
	# CPU FALLBACK ONLY FOR SMOKE TESTS (NO bfloat16 AUTOCAST)
	CUDA_DEV   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

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


def get_mean_stdd(HP,checkpoint_path):
	'''
	Normalization the model was trained with: from HP['train_set'] (absent for
	train.py rows = s2dw), unless the row sets mean_stdd (fine-tuned models keep
	their pretraining stats). Checked against the checkpoint's mean_stdd if stored.
	'''
	train_set = HP.get('train_set')
	assert train_set in TRAIN_SET_STATS, f"Unknown train_set '{train_set}' in id {HP['id']}"
	mean_stdd = HP.get('mean_stdd',TRAIN_SET_STATS[train_set])
	assert mean_stdd in dataloader.S1F11_MEAN_STDD, f"Unknown mean_stdd '{mean_stdd}' in id {HP['id']}"

	ckpt_norm = torch.load(checkpoint_path,map_location='cpu').get('mean_stdd')
	assert ckpt_norm in (None,mean_stdd), \
		f"{checkpoint_path} was trained with mean_stdd '{ckpt_norm}', hyperparameters give '{mean_stdd}'"
	return mean_stdd


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
	Pixels with T == IGNORE go to an extra bin that is dropped (S2DW masks have none).
	'''
	# [0,1,2,3] = [TN,FP,FN,TP], n_classes**2 = IGNORED
	nn   = n_classes*n_classes
	idx  = torch.where(T == IGNORE,nn,T*n_classes + Y).flatten().to(torch.int64)
	binc = torch.bincount(idx,minlength=nn+1)[:nn]
	confmat += binc.view(n_classes,n_classes)


def total_time_decorator(orig_func):
	@wraps(orig_func)
	def wrapper(*args, **kwargs):
		total_time_start = time.perf_counter()
		orig_func(*args,**kwargs)
		total_time = time.perf_counter() - total_time_start
		print(f'TOTAL TEST TIME: {total_time:.2f}s')
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
	print('\t'.join(header))
	print('\t'.join(result))
	print(f"Result file written to {path}")


def load_checkpoint(path, model):
	'''
	Loads model state from a .pth.tar checkpoint.
	'''
	# LOAD DICT/TENSORS TO DEV
	checkpoint = torch.load(path, map_location=CUDA_DEV)

	# LOAD WEIGHTS -- STRIP torch.compile PREFIX IF SAVED COMPILED
	state = {k.removeprefix('_orig_mod.'): v for k,v in checkpoint['model_state_dict'].items()}
	model.load_state_dict(state)

	# LOAD OTHER
	epoch = checkpoint.get('epoch')
	t_loss = checkpoint.get('t_loss')
	v_loss = checkpoint.get('v_loss')

	return model,epoch,t_loss,v_loss


def adjust_bands(bands,mean,std):

	if bands.shape[0] == 3:
		mean = mean[0:3]
		std  = std[0:3]

	# undo normalization in the GPU transform
	raw = bands * std.reshape(-1,1,1) + mean.reshape(-1,1,1)

	# resize and return
	raw = np.clip(raw,0,255).astype(np.uint8)
	return raw[0:3].transpose(1,2,0) # CHW -> HWC FOR PIL


def colorize_mask(mask,n_classes=2):
	'''
	0 black, 1 white, IGNORE (-1) gray.
	'''
	gray = np.where(mask == IGNORE,127,mask*255).astype(np.uint8)
	return np.repeat(gray[...,np.newaxis],3,axis=-1)


def colorize_errors(label,pred):
	'''
	False Positive: red
	False Negative: blue
	'''
	h,w = label.shape
	rgb = np.zeros((h,w,3),dtype=np.uint8)
	tp = (label==1) & (pred==label)
	fp = (label==0) & (pred==1)
	fn = (label==1) & (pred==0)

	rgb[tp] = (255,255,255)
	rgb[fp] = (255,0,0)
	rgb[fn] = (0,0,255)

	return rgb


def save_prediction_images(model,dataloader,aug,test_set,n_classes=2,n_images=30):

	img_dir = f"{LOG_DIR}/predicted_{model.model_id:03}_{test_set}"
	os.makedirs(img_dir,exist_ok=True)
	saved = 0

	# DENORMALIZE WITH THE TRANSFORM'S OWN STATS
	mean = aug.mean.cpu().numpy()
	std  = aug.stdd.cpu().numpy()

	model.eval()
	with torch.no_grad():
		for i,(X,T) in enumerate(dataloader):

			if saved >= n_images:
				break

			# TO GPU
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			X,T = aug(X,T)

			# FORWARD
			with torch.autocast(device_type=CUDA_DEV.type,dtype=torch.bfloat16,enabled=CUDA_DEV.type=='cuda'):
				output        = model(X)
				Y_soft,Y_hard = torch.max(output,1)


			X_cpu = X.float().cpu().numpy()
			T_cpu = T.cpu().numpy()
			Y_cpu = Y_hard.cpu().numpy()

			for b in range(X_cpu.shape[0]):

				if saved >= n_images:
					break

				bands = X_cpu[b,:,:,:]
				label = T_cpu[b]
				pred  = Y_cpu[b]
				bands = adjust_bands(bands,mean,std)
				label = colorize_mask(label)
				pred  = colorize_mask(pred)

				Image.fromarray(bands).save(f"{img_dir}/{saved:03}_bands.png")
				Image.fromarray(label).save(f"{img_dir}/{saved:03}_label.png")
				Image.fromarray(pred).save(f"{img_dir}/{saved:03}_preds.png")

				saved += 1

	print(f"{saved} predictions written to {img_dir}")


@total_time_decorator
def test(model,dataloader,aug,device,n_classes,test_set):

	# Count samples in GPU
	sample_sum = torch.zeros(1,device=device)
	gpu_mat    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	log_path   = f"{LOG_DIR}/test_{model.model_id:03}_{test_set}.tsv"

	# PASS TEST SET
	model.eval()
	with torch.no_grad():
		for X,T in dataloader:

			# TO GPU
			X = X.to(device,non_blocking=True)
			T = T.to(device,non_blocking=True)

			X,T = aug(X,T) # NORMALIZE; 0/255 -> 0/1 (S1F11: 127 -> IGNORE)

			# FORWARD
			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
				output        = model(X)
				Y_soft,Y_hard = torch.max(output,1)

			# APPEND COUNTS
			sample_sum += X.size(0)
			update_confusion_matrix(gpu_mat,T,Y_hard,n_classes)

	# GET METRICS OVER ENTIRE SET
	cpu_mat = gpu_mat.cpu() #syncs gpu-cpu
	ppv,tpr,acc,iou = calculate_metrics(cpu_mat)
	print(f"{int(sample_sum.item())} tiles, {int(cpu_mat.sum())} labeled pixels")

	# LOG RESULT
	results = {'_acc':acc,'_tpr':tpr,'_ppv':ppv,'_iou':iou} #irrespective order
	log_results(log_path,results)


if __name__ == '__main__':

	# LOAD SCRIPT PARAMETERS
	args = parse_args()

	# LOAD HYPERPARAMETER DICT
	with open(args.params,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line.strip()]
	hp_list_indexed = {row['id']:row for row in HP_LIST}

	checkpoint_path = args.checkpoint
	model_id = int(checkpoint_path.split('/')[-1].split('_')[1])

	print(f"Loading checkpoint: {checkpoint_path}")

	# LOAD HYPERPARAMETERS
	assert model_id in hp_list_indexed, f"MODEL ID '{model_id}' NOT IN HYPERPARAMETER FILE."
	HP = hp_list_indexed[model_id]
	n_bands    = 4
	n_classes  = 2
	batch_size = HP['batch']
	mean_stdd  = get_mean_stdd(HP,checkpoint_path)

	print(f"Hyperparameters:\n\n{HP}\n")
	print(f"Test set: {args.test_set} | normalization: {mean_stdd} | device: {CUDA_DEV}")
	if args.test_set == 's2dw': # GPUValidationTransform HAS THE S2DW STATS BUILT IN
		assert mean_stdd == 's2dw', \
			f"GPUValidationTransform normalizes with the S2DW stats; this model was trained with '{mean_stdd}'"

	# LOAD CHECKPOINT WEIGHTS -- SAME CONSTRUCTION AS train.py / train_s1f11.py
	net = models.UNet(
		HP['id'],
		encoder=HP['encoder'],
		decoder=HP['decoder'],
		in_channels=n_bands,
		out_labels=n_classes,
		cnn_layers=HP['cnn_layers'],
		vit_layers=HP['vit_layers'],
		init_channels=HP['init_chan']
	)
	net,_,_,_ = load_checkpoint(checkpoint_path,net)
	net = net.to(CUDA_DEV)

	# DATASET + GPU TRANSFORM
	bands_np,masks_np = dataloader.load_cached_split(DATA_DIR,split=TEST_CACHE[args.test_set])
	if args.test_set == 's2dw':
		dataset = dataloader.SentinelFastDataset(bands_np,masks_np,n_bands=n_bands)
		aug     = dataloader.GPUValidationTransform().to(CUDA_DEV)
	else:
		dataset = dataloader.S1F11FastDataset(bands_np,masks_np)
		aug     = dataloader.S1F11GPUValidationTransform(mean_stdd=mean_stdd).to(CUDA_DEV)

	test_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		drop_last=False,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=10 if args.num_workers > 0 else None
	)

	if args.save_images:
		print("Saving prediction masks")
		save_prediction_images(net,test_loader,aug,args.test_set,n_classes)
		sys.exit(0)

	# RUN TEST
	test(net,test_loader,aug,CUDA_DEV,n_classes,args.test_set)
