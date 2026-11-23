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


def sample(spec):
    '''
    ('log',lo,hi)   -> log-uniform in [lo,hi]
    ('choice',[..]) -> uniform over the list
    '''
    if spec[0] == 'log':
        _, lo, hi = spec
        return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
    return spec[1][np.random.randint(len(spec[1]))]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_experiments(n=30,seed=476,path='./models/hparams/experiments_1.json'):

	# SAMPLING SPACE
	space = {
		'lrate': ('log', 5e-5, 5e-4),
		'decay': ('log', 1e-3, 1e-1)
	}


	# FIXED PARAMETERS
	batch      = 32
	epochs     = 100

	# SET GRID
	cnn_layers = [2,3]
	block_chan = [32]
	grid = []
	for encoder_type, decoder_type in itertools.product(['cnn', 'vit'], ['cnn', 'vit']):
		vit_layers = [0] if encoder_type == decoder_type == 'cnn' else [1, 2]

		for n_vit,n_cnn,init_chan in itertools.product(vit_layers,cnn_layers,block_chan):
			grid.append((encoder_type,decoder_type,init_chan,n_cnn,n_vit))

	# SHARED DRAWS -- SAME n (lrate,decay) PAIRS FOR EVERY GRID CONFIG
	draws = []
	for _ in range(n):
		d = {k: sample(v) for k, v in space.items()}
		for k in ('lrate', 'decay'):
			d[k] = float(f"{d[k]:.3g}")
		draws.append(d)

	i=0
	with open(path,'w') as fp:

		for encoder_type,decoder_type,init_chan,n_cnn,n_vit in grid:
			for d in draws:

				exp = dict(d)

				exp['batch'] = batch
				exp['encoder'] = encoder_type
				exp['decoder'] = decoder_type
				exp['init_chan'] = init_chan
				exp['cnn_layers'] = n_cnn
				exp['vit_layers'] = n_vit

				exp.update({'id': i, 'seed': seed, 'epochs': epochs})
				fp.write(json.dumps(exp) + '\n')
				i += 1


def set_seed_experiments(n=20,best=(0,1,2,3),path='./hparams/experiments_2.json'):
	'''
	Train the best model configuration for ViT-ViT, CNN-CNN, CNN-ViT, and ViT-CNN
	over 'n' distinct seeds.
	'''
	seeds = list(range(476,476+n,1))
	pass


HPARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'hparams') # models/hparams, ANY CWD

# (epochs, warmup_epochs) PER TRAINING SET, BATCH 32 AS IN experiments_1 (S2DW: 100, 10 =
# 45,800 STEPS). ROUND NUMBERS NEAR S2DW'S STEP COUNT: hand 30 STEPS/EPOCH x 1500 = 45,000,
# weak 517 x 100 = 51,700. WARMUP IS 10% OF EPOCHS IN ALL THREE.
S1F11_EPOCHS = {
	'hand': (1500,150),
	'weak': (100,10)}


ARCH_KEYS = ('encoder','decoder','init_chan','cnn_layers','vit_layers') # ONE ARCHITECTURE OF THE SWEEP


def set_s1f11_experiments(train_set,name,arch_id,source='experiments_1.json',out_dir=HPARAMS_DIR,
		mean_stdd=None,init_checkpoint=None):
	'''
	Copies the rows of the S2DW sweep (source, in hparams/) that share the architecture
	(ARCH_KEYS) of row arch_id -- the n (lrate, decay) draws of that architecture, 40 in
	experiments_1 -- to {out_dir}/{name}.json for train_s1f11.py: same id, architecture,
	(lrate, decay) draw, batch and seed, plus train_set and mean_stdd (from scratch, the
	training set's own stats). Each dataset gets the same search (same draws, same budget)
	and picks its own best on its own validation set.
	arch_id: the best experiments_1 id (selected on S2DW validation, like arm A), or a
	list of ids to keep several architectures (e.g. the best of each encoder-decoder family).
	mean_stdd, init_checkpoint: for fine-tuning (set_pretrained_experiments); by default
	the rows train from scratch with mean_stdd = train_set.
	epochs and warmup_epochs come from S1F11_EPOCHS: hand 1500 (150 warmup), weak 100
	(10 warmup), vs S2DW 100 (10).
	No 'k' (train_s1f11.py rejects it) and no init_checkpoint (models train from scratch).
	Ids repeat across files: write each file's checkpoints/logs to their own directory.
	'''
	assert train_set in ('hand','weak'), f"train_set must be 'hand' or 'weak', got {train_set}"
	assert (mean_stdd is None) == (init_checkpoint is None), "mean_stdd and init_checkpoint go together (fine-tuning)"
	mean_stdd = mean_stdd or train_set
	with open(os.path.join(HPARAMS_DIR,source),'r') as fp:
		base = [json.loads(line) for line in fp if line.strip()]
	by_id = {r['id']: r for r in base}
	assert len(by_id) == len(base), f"duplicate ids in {source}"

	# ARCHITECTURES TO KEEP
	arch_ids = [arch_id] if isinstance(arch_id,int) else list(arch_id)
	assert arch_ids, "arch_id is empty"
	for i in arch_ids:
		assert i in by_id, f"arch_id {i} not in {source}"
	archs = {tuple(by_id[i][k] for k in ARCH_KEYS) for i in arch_ids}

	rows = []
	for r in base:
		if tuple(r[k] for k in ARCH_KEYS) not in archs:
			continue
		assert 'train_set' not in r and 'mean_stdd' not in r, f"{source} id {r['id']} is not an S2DW (train.py) row"
		epochs,warmup = S1F11_EPOCHS[train_set]
		row = {**r,'epochs': epochs,'warmup_epochs': warmup,'train_set': train_set,'mean_stdd': mean_stdd}
		if init_checkpoint is not None:
			row['init_checkpoint'] = init_checkpoint
		rows.append(row)

	for a in sorted(archs):
		n = sum(tuple(r[k] for k in ARCH_KEYS) == a for r in rows)
		print(f"[{name}] {dict(zip(ARCH_KEYS,a))}: {n} rows")
	write_hp_file(name,rows,out_dir)
	return rows


def set_s2weak_experiments(arch_id,name='experiments_s2weak',source='experiments_1.json'):
	'''
	Train UNet models on the 'S2 weakly labeled' data of Sen1Floods11 (arm C: S2Weak +
	S2IndexLabelWeak, Bolivia removed), from scratch: the architecture of source row
	arch_id with its (lrate, decay) draws from the S2DW sweep.
	'''
	return set_s1f11_experiments('weak',name,arch_id,source)


def set_s2hand_experiments(arch_id,name='experiments_s2hand',source='experiments_1.json'):
	'''
	Train UNet models on the hand-labeled training data of Sen1Floods11 (arm B, all
	252 'train' chips), from scratch: the architecture of source row arch_id with its
	(lrate, decay) draws from the S2DW sweep. k-chip subsets are trained by a separate
	loop in train_s1f11.py.
	'''
	return set_s1f11_experiments('hand',name,arch_id,source)


PRETRAIN_STATS = { # PRETRAINING DATA -> NORMALIZATION ITS CHECKPOINTS WERE TRAINED WITH
	's2dw': 's2dw',     # train.py (experiments_1)
	'weak': 'weak'}     # train_s1f11.py (experiments_s2weak)


def set_pretrained_experiments(arch_id,pretrain,init_checkpoint,finetune='hand',name=None,source='experiments_1.json'):
	'''
	Fine tune on hand-labeled data (or weakly-labeled S2) starting from
	a previously trained S2DW or S2Weak checkpoint.

	Pre    -> Fine-tune    -> Test
	--------------------------------------
	S2DW   -> Hand-labeled -> Hand-labeled   arm D:          pretrain='s2dw', finetune='hand'
	S2Weak -> Hand-labeled -> Hand-labeled   C, fine-tuned:  pretrain='weak', finetune='hand'
	S2DW   -> S2Weak       -> Hand-labeled   (optional)      pretrain='s2dw', finetune='weak'

	Same rows as the from-scratch sweep on the fine-tuning data (set_s2hand_experiments /
	set_s2weak_experiments with the same arch_id): same ids, architecture, (lrate, decay)
	draws, epochs, batch and seed, so fine-tuned and from-scratch runs differ only in
	the initial weights. Adds init_checkpoint and mean_stdd = the pretraining's stats
	(train_s1f11.py checks them against the checkpoint).
	arch_id         : the architecture (an experiments_1 id), as in the other sweeps.
	init_checkpoint : the pretrained best_{id:03}_e{epoch:02}.pth.tar as the training pod sees
	                  it (e.g. /model_pvc/experiments_1/best_007_e42.pth.tar). Its id must
	                  have arch_id's architecture.
	name            : default experiments_{s2dw|s2weak}_{s2hand|s2weak}.
	'''
	assert pretrain in PRETRAIN_STATS, f"pretrain must be one of {list(PRETRAIN_STATS)}, got {pretrain}"
	assert finetune in ('hand','weak'), f"finetune must be 'hand' or 'weak', got {finetune}"
	assert pretrain != finetune, f"pretrain and finetune are both '{pretrain}'"
	assert isinstance(arch_id,int), "one architecture: the pretrained checkpoint has a single one"
	assert init_checkpoint, "init_checkpoint is required"

	# THE CHECKPOINT'S RUN (ID IN ITS NAME, SAME IDS IN ALL SWEEP FILES) MUST HAVE arch_id's ARCHITECTURE
	with open(os.path.join(HPARAMS_DIR,source),'r') as fp:
		by_id = {r['id']: r for r in (json.loads(line) for line in fp if line.strip())}
	assert arch_id in by_id, f"arch_id {arch_id} not in {source}"
	try:
		ckpt_id = int(os.path.basename(init_checkpoint).split('_')[1]) # best_007_e42.pth.tar -> 7
	except (IndexError,ValueError):
		ckpt_id = None
		print(f"WARNING: no run id in {init_checkpoint} (expected best_{{id:03}}_e{{epoch:02}}.pth.tar); architecture not checked")
	if ckpt_id is not None:
		assert ckpt_id in by_id, f"{init_checkpoint}: id {ckpt_id} not in {source}"
		arch = lambda i: {k: by_id[i][k] for k in ARCH_KEYS}
		assert arch(ckpt_id) == arch(arch_id), \
			f"{init_checkpoint} (id {ckpt_id}) is {arch(ckpt_id)}, arch_id {arch_id} is {arch(arch_id)}"

	short = {'s2dw': 's2dw','hand': 's2hand','weak': 's2weak'}
	name  = name or f"experiments_{short[pretrain]}_{short[finetune]}"
	return set_s1f11_experiments(finetune,name,arch_id,source,
		mean_stdd=PRETRAIN_STATS[pretrain],init_checkpoint=init_checkpoint)


def write_hp_file(name,rows,hp_dir='./hparams'):
	# WRITE JSON FILE
	out_file_path = f"{hp_dir}/{name}.json"		
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':
	# set_seed(476) #Set seed to fix list of hyperparameters

	# S2DW SWEEP (experiments_1.json). ITS DRAWS ARE NOT SEEDED (set_seed ABOVE IS
	# COMMENTED OUT): RERUNNING IT OVERWRITES THE BASELINE WITH DIFFERENT (lrate, decay)
	# set_experiments(n=40,seed=476)

	# SEN1FLOODS11 SWEEPS FOR train_s1f11.py -- THE (lrate, decay) DRAWS OF ONE experiments_1
	# ARCHITECTURE. SET BEST_ARCH_ID TO THE BEST experiments_1 ID (ON S2DW VALIDATION)
	BEST_ARCH_ID = None
	if BEST_ARCH_ID is None:
		print("Set BEST_ARCH_ID in experiments.py (best experiments_1 id) to write the S2Weak/S2Hand files.")
	else:
		set_s2weak_experiments(BEST_ARCH_ID)
		set_s2hand_experiments(BEST_ARCH_ID)

	# # FINE-TUNING SWEEPS -- PRETRAINED CHECKPOINTS AS THE TRAINING POD SEES THEM. S2DW: THE BEST
	# # experiments_1 RUN (= BEST_ARCH_ID'S BEST EPOCH). S2WEAK: THE BEST experiments_s2weak RUN.
	# S2DW_CHECKPOINT   = None # e.g. '/model_pvc/experiments_1/best_007_e42.pth.tar'
	# S2WEAK_CHECKPOINT = None # e.g. '/model_pvc/experiments_s2weak/best_012_e88.pth.tar'
	# if BEST_ARCH_ID is not None and S2DW_CHECKPOINT is not None:
	# 	set_pretrained_experiments(BEST_ARCH_ID,'s2dw',S2DW_CHECKPOINT,finetune='hand')   # ARM D
	# if BEST_ARCH_ID is not None and S2WEAK_CHECKPOINT is not None:
	# 	set_pretrained_experiments(BEST_ARCH_ID,'weak',S2WEAK_CHECKPOINT,finetune='hand') # C, FINE-TUNED
