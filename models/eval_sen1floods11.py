'''
Zero-shot evaluation of an S2DW-trained model on the hand-labeled Sen1Floods11
chips (Bonafilia et al., 2020), with NDWI baselines on the same pixels.

Sen1Floods11 S2Hand chips are Sentinel-2 Level-1C, 13 bands, TOA reflectance x 10000,
512x512 at 10 m; LabelHand is -1 (not valid) / 0 (not water) / 1 (water). Each
chip is prepared to look like an S2DW chip:
	1. keep B04,B03,B02,B08 (R,G,B,NIR -- the S2DW band order)
	2. add --dn-offset (default 1000): S2DW stretched raw 2025 L2A DNs, which carry
	   a +1000 offset; Sen1Floods11 (2016-2019 L1C) has none
	3. clip each band at its 99th percentile over valid pixels, scale to 0-255
	   (chip.py does this per Sentinel-2 scene; here it can only be per chip)
	4. normalize with the S2DW training mean/std, split into four 256x256 tiles
Remaining differences: L1C (top of atmosphere) vs L2A (surface reflectance), per-chip
vs per-scene stretch, and flood vs permanent water labels. S2Hand has no cloud mask;
the per-chip 'bright_frac' column (blue TOA >= 0.2) is a rough proxy for filtering.

Scored pixels: label != -1 and all four bands non-zero. Methods:
	model           the checkpoint, argmax over 2 classes
	ndwi>=0         NDWI = (B3-B8)/(B3+B8) on TOA reflectance, fixed threshold
	ndwi>=<t>       same, with --ndwi-threshold (e.g. the value fit on S2DW)
	ndwi-otsu       Otsu threshold on each chip's NDWI histogram (uses no labels)
IoU is pooled over all scored pixels of a split (as in models/train.py); the
mean of per-chip IoUs (chips where it is defined) is reported alongside.
Results are also grouped by event and by the chip's labeled water fraction
(<5%, 5-25%, 25-75%, >75%). S2DW chips are all 25-75% water, so that group is the
like-for-like comparison; most Sen1Floods11 chips are almost all land.

Data (only S2Hand, LabelHand and the split CSVs are needed, ~1 GB):
	gsutil -m rsync -r gs://sen1floods11/v1.1/splits <root>/v1.1/splits
	gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand <root>/v1.1/data/flood_events/HandLabeled/S2Hand
	gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand <root>/v1.1/data/flood_events/HandLabeled/LabelHand

Usage (from models/):
	python3 eval_sen1floods11.py --s1f11-dir <root> --checkpoint <net-dir>/best_007_e42.pth.tar \
		-p hparams/experiments_1.json [--splits test bolivia] [--ndwi-threshold T] [--out-dir DIR]
'''
import os
import json
import argparse

import numpy as np
import torch
import tifffile

import models
import dataloader

################################################################################
# SETTINGS
################################################################################
S2_RGBN  = [3,2,1,7]      # S2Hand BAND INDICES OF B04,B03,B02,B08 -- S2DW ORDER R,G,B,NIR
TILE     = 256            # S2DW CHIP SIZE
N_BINS   = 2000           # NDWI HISTOGRAM BINS OVER [-1,1]
BRIGHT   = 2000           # BLUE TOA x 10000 AT/ABOVE WHICH A PIXEL COUNTS AS 'BRIGHT'

# WATER-FRACTION GROUPS (FRACTION OF SCORED PIXELS LABELED WATER). S2DW KEEPS ONLY
# CHIPS WITH 25-75% WATER (chip.py WATER_MIN/WATER_MAX, BOTH INCLUSIVE), SO
# 'water25-75%' IS THE LIKE-FOR-LIKE GROUP AND THE OTHERS SHOW WHAT THAT FILTER COSTS.
WATER_BINS = [
	('water<5%',    lambda f: f < 0.05),
	('water5-25%',  lambda f: 0.05 <= f < 0.25),
	('water25-75%', lambda f: 0.25 <= f <= 0.75),
	('water>75%',   lambda f: f > 0.75)]
SPLIT_CSV = {
	'train':   'flood_train_data.csv',
	'valid':   'flood_valid_data.csv',
	'test':    'flood_test_data.csv',
	'bolivia': 'flood_bolivia_data.csv'}

################################################################################
# DATA
################################################################################
def chip_ids(root,split):
	'''
	Chip ids (e.g. Ghana_313799) from a hand-labeled split CSV. The CSVs have
	CRLF line endings and list the S1Hand file first.
	'''
	path = f"{root}/splits/flood_handlabeled/{SPLIT_CSV[split]}"
	with open(path,'r') as fp:
		rows = [line.strip() for line in fp if line.strip()]
	return [r.split(',')[0].replace('_S1Hand.tif','') for r in rows]


def read_chip(root,cid):
	'''
	Returns S2Hand as int16 [13,H,W] and LabelHand as int16 [H,W].
	'''
	base = f"{root}/data/flood_events/HandLabeled"
	s2   = tifffile.imread(f"{base}/S2Hand/{cid}_S2Hand.tif")
	if s2.shape[0] != 13: # PIXEL-INTERLEAVED FILE -> BAND FIRST
		s2 = np.moveaxis(s2,-1,0)
	lbl  = tifffile.imread(f"{base}/LabelHand/{cid}_LabelHand.tif")
	return s2, np.squeeze(lbl)


def to_s2dw_bytes(s2,valid,dn_offset):
	'''
	R,G,B,NIR as uint8 [4,H,W], stretched like chip.py: clip at the 99th
	percentile of the valid pixels, scale to 0-255. Invalid pixels are 0.
	'''
	out = np.zeros((4,*s2.shape[1:]),np.uint8)
	for c,b in enumerate(S2_RGBN):
		x  = s2[b].astype(np.float64) + dn_offset
		hi = int(np.percentile(x[valid],99))
		if hi <= 0:
			continue
		out[c] = np.round(np.clip(x,0,hi) / hi * 255).astype(np.uint8)
	out[:,~valid] = 0
	return out

################################################################################
# PREDICTION
################################################################################
def load_model(args,device):
	'''
	Same construction as models/train.py, weights from a train.py checkpoint.
	'''
	with open(args.params,'r') as fp:
		rows = {r['id']: r for r in (json.loads(l) for l in fp if l.strip())}
	assert args.id in rows, f"id {args.id} not in {args.params}"
	HP  = rows[args.id]
	net = models.UNet(
		HP['id'],
		encoder=HP['encoder'],
		decoder=HP['decoder'],
		in_channels=args.bands,
		out_labels=2,
		cnn_layers=HP['cnn_layers'],
		vit_layers=HP['vit_layers'],
		init_channels=HP['init_chan'],
		mlp_ratio=HP.get('mlp_ratio',4))
	ckpt  = torch.load(args.checkpoint,map_location='cpu')
	state = {k.removeprefix('_orig_mod.'): v for k,v in ckpt['model_state_dict'].items()}
	net.load_state_dict(state)
	print(f"Model: id {HP['id']} {HP['encoder']}-{HP['decoder']} cnn={HP['cnn_layers']} vit={HP['vit_layers']} "
		f"C={HP['init_chan']} | {args.checkpoint} | device {device}")
	return net.to(device).eval()


@torch.inference_mode()
def predict(net,img,device,n_bands):
	'''
	img: uint8 [4,H,W] (H,W multiples of TILE). Returns the class map [H,W].
	'''
	_,H,W = img.shape
	tiles = [img[:n_bands,i:i+TILE,j:j+TILE] for i in range(0,H,TILE) for j in range(0,W,TILE)]
	x = torch.from_numpy(np.stack(tiles)).to(device).float()
	x = (x - dataloader.MEAN[:n_bands].to(device)) / dataloader.STDD[:n_bands].to(device)
	with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
		y = net(x).argmax(dim=1).cpu().numpy().astype(np.uint8)
	out = np.zeros((H,W),np.uint8)
	k = 0
	for i in range(0,H,TILE):
		for j in range(0,W,TILE):
			out[i:i+TILE,j:j+TILE] = y[k]
			k += 1
	return out

################################################################################
# NDWI + METRICS
################################################################################
def ndwi_toa(s2):
	'''
	NDWI on L1C TOA reflectance (no offset before processing baseline 04.00).
	'''
	g = s2[2].astype(np.float32); n = s2[7].astype(np.float32)
	den = g + n
	return np.divide(g-n,den,out=np.zeros_like(den),where=den > 0)


def otsu_threshold(values):
	'''
	Otsu threshold on NDWI values in [-1,1].
	'''
	hist,edges = np.histogram(values,bins=N_BINS,range=(-1,1))
	p   = hist.astype(np.float64); mid = np.arange(N_BINS) + 0.5
	w0  = np.cumsum(p)[:-1]; w1 = p.sum() - w0
	m0  = np.cumsum(p*mid)[:-1]; m1 = (p*mid).sum() - m0
	ok  = (w0 > 0) & (w1 > 0)
	between = np.zeros_like(w0)
	between[ok] = w0[ok]*w1[ok] * (m0[ok]/w0[ok] - m1[ok]/w1[ok])**2
	return float(edges[int(np.argmax(between)) + 1])


def confusion(pred,water):
	'''
	(TP,FP,FN,TN) for boolean arrays over the scored pixels.
	'''
	tp = int((pred & water).sum()); fp = int((pred & ~water).sum())
	fn = int((~pred & water).sum()); tn = int((~pred & ~water).sum())
	return np.array([tp,fp,fn,tn],np.int64)


def metrics(c):
	'''
	Same definitions as calculate_metrics() in models/train.py.
	'''
	tp,fp,fn,tn = (int(v) for v in c)
	d = lambda a,b: a/b if b > 0 else float('nan')
	return {'iou1': d(tp,tp+fp+fn), 'iou0': d(tn,tn+fn+fp),
		'ppv1': d(tp,tp+fp), 'tpr1': d(tp,tp+fn), 'acc': d(tp+tn,tp+fp+fn+tn)}


def water_bin(frac):
	'''
	Name of the WATER_BINS group a chip's water fraction falls in.
	'''
	return next(name for name,test in WATER_BINS if test(frac))

################################################################################
# EVALUATION + OUTPUT
################################################################################
def evaluate(net,args):
	'''
	Scores the model and the NDWI baselines on every chip of args.splits.
	Returns the per-chip records, the pooled confusions keyed by
	(split, group, method) with group = 'all', an event or a water bin, and the
	list of methods.
	'''
	methods = ['model','ndwi>=0'] + ([f'ndwi>={args.ndwi_threshold:g}'] if args.ndwi_threshold is not None else []) + ['ndwi-otsu']
	pooled  = {}
	records = []

	for split in args.splits:
		ids = chip_ids(args.s1f11_dir,split)
		print(f"[{split}] {len(ids)} chips",flush=True)
		for cid in ids:
			s2,lbl = read_chip(args.s1f11_dir,cid)
			valid  = (lbl != -1) & np.all(s2[S2_RGBN] > 0,axis=0)
			water  = lbl == 1
			event  = cid.split('_')[0]
			rec    = {'split': split, 'chip': cid, 'event': event,
				'valid_frac': float(valid.mean()),
				'water_frac': float(water[valid].mean()) if valid.any() else float('nan'),
				'bright_frac': float((s2[1][valid] >= BRIGHT).mean()) if valid.any() else float('nan')}
			if valid.sum() < 100: # NOTHING MEANINGFUL TO SCORE
				rec['status'] = 'no-valid-pixels'
				records.append(rec)
				continue

			img  = to_s2dw_bytes(s2,valid,args.dn_offset)
			ndwi = ndwi_toa(s2)
			preds = {'model': predict(net,img,args.device,args.bands) == 1,
				'ndwi>=0': ndwi >= 0}
			if args.ndwi_threshold is not None:
				preds[f'ndwi>={args.ndwi_threshold:g}'] = ndwi >= args.ndwi_threshold
			t_otsu = otsu_threshold(ndwi[valid])
			preds['ndwi-otsu'] = ndwi >= t_otsu
			wbin = water_bin(rec['water_frac'])
			rec.update({'status': 'ok', 'otsu_t': t_otsu, 'water_bin': wbin})

			for m in methods:
				c = confusion(preds[m][valid],water[valid])
				for key in ((split,'all',m),(split,event,m),(split,wbin,m)):
					pooled[key] = pooled.get(key,np.zeros(4,np.int64)) + c
				rec[f'iou1:{m}'] = metrics(c)['iou1']
			records.append(rec)

	return records,pooled,methods


def summarize(records,pooled,methods,splits):
	'''
	One row per (split, group, method): pooled metrics + mean of per-chip IoU.
	Groups: 'all', each water-fraction bin, each event.
	'''
	rows = []
	for split in splits:
		groups = ['all'] + [b for b,_ in WATER_BINS] + sorted({r['event'] for r in records if r['split'] == split})
		for g in groups:
			recs = [r for r in records if r['split'] == split and r['status'] == 'ok'
				and g in ('all',r['event'],r['water_bin'])]
			for m in methods:
				if (split,g,m) not in pooled:
					continue
				per_chip = [r[f'iou1:{m}'] for r in recs if not np.isnan(r[f'iou1:{m}'])]
				rows.append({'split': split, 'group': g, 'method': m, 'chips': len(recs),
					'pixels': int(pooled[(split,g,m)].sum()), **metrics(pooled[(split,g,m)]),
					'chip_mean_iou1': float(np.mean(per_chip)) if per_chip else float('nan')})
	return rows


def write_tsv(path,rows,cols):
	fmt = lambda v: f"{v:.5f}" if isinstance(v,float) else str(v)
	with open(path,'w') as fp:
		fp.write('\t'.join(cols)+'\n')
		for r in rows:
			fp.write('\t'.join(fmt(r.get(c,'')) for c in cols)+'\n')


def write_results(out_dir,rows,records,methods):
	'''
	s1f11_summary.tsv (summarize() rows) and s1f11_chips.tsv (one row per chip).
	'''
	write_tsv(f"{out_dir}/s1f11_summary.tsv",rows,
		['split','group','method','chips','pixels','iou1','iou0','ppv1','tpr1','acc','chip_mean_iou1'])
	write_tsv(f"{out_dir}/s1f11_chips.tsv",records,
		['split','chip','event','status','valid_frac','water_frac','water_bin','bright_frac','otsu_t'] + [f'iou1:{m}' for m in methods])


def print_results(rows,records,out_dir):
	'''
	Stdout: split totals, then the water-fraction groups.
	'''
	header = f"{'split':<9}{'group':<13}{'method':<14}{'chips':>6}{'IoU_1':>9}{'IoU_0':>9}{'PPV_1':>9}{'TPR_1':>9}{'chipIoU':>9}"
	for title,keep in (("ALL CHIPS",lambda g: g == 'all'),
			("BY WATER FRACTION (S2DW chips are all 25-75%)",lambda g: g.startswith('water'))):
		print(f"\n{title}\n{header}")
		for r in rows:
			if keep(r['group']):
				print(f"{r['split']:<9}{r['group']:<13}{r['method']:<14}{r['chips']:>6}{r['iou1']:>9.4f}{r['iou0']:>9.4f}"
					f"{r['ppv1']:>9.4f}{r['tpr1']:>9.4f}{r['chip_mean_iou1']:>9.4f}")
	skipped = sum(r['status'] != 'ok' for r in records)
	if skipped:
		print(f"\n{skipped} chip(s) had no valid pixels and were skipped.")
	print(f"\nPer-event results in {out_dir}/s1f11_summary.tsv; per-chip in {out_dir}/s1f11_chips.tsv")

################################################################################
# MAIN
################################################################################
def parse_args():
	parser = argparse.ArgumentParser(description="Zero-shot S2DW model + NDWI evaluation on Sen1Floods11 hand labels.")
	parser.add_argument('--s1f11-dir',required=True,help="Sen1Floods11 root (the v1.1 folder, or its parent).")
	parser.add_argument('--checkpoint',required=True,help="train.py checkpoint (.pth.tar).")
	parser.add_argument('-p','--params',required=True,help="Hyperparameter JSON lines file the model was trained from.")
	parser.add_argument('--id',type=int,default=None,help="Row id in --params (default: parsed from the checkpoint name).")
	parser.add_argument('--bands',type=int,default=4,choices=[3,4],help="Model input bands: 4 = RGB+NIR (train.py), 3 = RGB.")
	parser.add_argument('--splits',nargs='+',default=['test','bolivia'],choices=list(SPLIT_CSV),
		help="Hand-labeled splits to evaluate (default: test bolivia). The model never saw any of them.")
	parser.add_argument('--ndwi-threshold',type=float,default=None,help="Extra NDWI threshold to report (e.g. fit on S2DW).")
	parser.add_argument('--dn-offset',type=float,default=1000,help="Added to S2Hand DNs before the stretch (default 1000, see docstring).")
	parser.add_argument('--out-dir',default='./s1f11_eval',help="Output directory for the .tsv files.")
	parser.add_argument('--device',default=None,help="torch device (default: cuda if available, else cpu).")
	args = parser.parse_args()

	root = args.s1f11_dir.rstrip('/')
	if os.path.isdir(f"{root}/v1.1"):
		root = f"{root}/v1.1"
	assert os.path.isdir(f"{root}/splits/flood_handlabeled"), f"No splits/flood_handlabeled under {root}"
	args.s1f11_dir = root

	assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
	assert os.path.isfile(args.params), f"Params file not found: {args.params}"
	if args.id is None: # best_007_e42.pth.tar / chkp_007_e42.pth.tar -> 7
		args.id = int(os.path.basename(args.checkpoint).split('_')[1])
	args.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
	os.makedirs(args.out_dir,exist_ok=True)
	return args


if __name__ == '__main__':
	args = parse_args()
	net  = load_model(args,args.device)
	records,pooled,methods = evaluate(net,args)
	rows = summarize(records,pooled,methods,args.splits)
	write_results(args.out_dir,rows,records,methods)
	print_results(rows,records,args.out_dir)
