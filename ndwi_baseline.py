'''
NDWI threshold baseline on the S2DW chips, computed from Sentinel-2 L2A surface
reflectance (not from the 8-bit chips, whose per-band 99th-percentile stretch
distorts the green/NIR ratio).

	NDWI = (G - NIR) / (G + NIR)       G = B03, NIR = B08 (McFeeters, 1996)
	water <=> NDWI >= t

For every chip in the evaluated splits, the matching 256x256 window is read from
the original B03/B08 .jp2 files, using the same alignment code as chip.py
(Product, get_strided_windows), and scored against the published chip label.
Each window is checked against the chip's own bands so a misaligned read cannot
pass silently.

Methods reported, per evaluated split (pixel counts pooled over all chips,
same IoU definition as models/train.py):
	ndwi>=0         fixed threshold, no fitting
	fit:<split>     one global threshold, chosen to maximise water IoU on --fit-split
	otsu-per-chip   Otsu threshold on each chip's NDWI histogram (uses no labels)
	oracle          best global threshold on the evaluated split itself (upper bound
	                for any single threshold, not a fair baseline)

Usage (from the repo root):
	python3 ndwi_baseline.py --s2-dir <dir with eodata/...> --dw-dir <DW .tif dir> \
		--chip-dir <chips>_sorted [--eval-split testing] [--fit-split validation] \
		[--out-dir ./ndwi_baseline] [--workers 4]

Outputs in --out-dir: ndwi_summary.tsv (one row per split x method) and
ndwi_chips.tsv (one row per chip).
'''
import os
import re
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# SAME ALIGNMENT/WINDOWING AS THE PUBLISHED CHIPS
from chip import Product, get_strided_windows, get_datastrip_id, get_dw_id

################################################################################
# SETTINGS
################################################################################
N_BINS   = 2000                                  # NDWI HISTOGRAM BINS OVER [-1,1] (WIDTH 0.001)
EDGES    = np.linspace(-1.0,1.0,N_BINS+1)        # CANDIDATE THRESHOLDS
MAX_RESID = 1.0                                  # ALIGNMENT CHECK: MAX |chip - fit(DN)| IN 8-BIT LEVELS
SAFE_GLOB = "eodata/Sentinel-2/MSI/L2A/*/*/*/*.SAFE" # SAME LAYOUT AS chip.py

################################################################################
# L2A RADIOMETRY
################################################################################
def l2a_radiometry(safe_path):
	'''
	Return {'B03': (offset,quant), 'B08': (offset,quant)} so that
	reflectance = (DN + offset) / quant.

	Read from MTD_MSIL2A.xml when present. The download queue fetches only the
	10 m .jp2 files, so otherwise fall back to the processing baseline in the
	.SAFE name: from baseline 04.00 (Jan 2022) on, L2A has BOA_ADD_OFFSET=-1000;
	before it, no offset. Quantification is 10000 in both cases.
	'''
	mtd = os.path.join(safe_path,"MTD_MSIL2A.xml")
	if os.path.isfile(mtd):
		root  = ET.parse(mtd).getroot()
		quant = None
		offsets  = {} # bandId -> offset
		physical = {} # 'B3' -> bandId
		for el in root.iter():
			tag = el.tag.split('}')[-1]
			if tag == 'BOA_QUANTIFICATION_VALUE':
				quant = float(el.text)
			elif tag == 'BOA_ADD_OFFSET':
				offsets[el.attrib['band_id']] = float(el.text)
			elif tag == 'Spectral_Information' and 'physicalBand' in el.attrib:
				physical[el.attrib['physicalBand']] = el.attrib['bandId']
		if quant is None:
			raise ValueError(f"No BOA_QUANTIFICATION_VALUE in {mtd}")
		ids = {'B03': physical.get('B3','2'), 'B08': physical.get('B8','7')}
		return {b: (offsets.get(i,0.0),quant) for b,i in ids.items()}

	m = re.search(r'_N(\d{4})_',os.path.basename(safe_path))
	if m is None:
		raise ValueError(f"No MTD_MSIL2A.xml and no processing baseline in {safe_path}")
	offset = -1000.0 if int(m.group(1)) >= 400 else 0.0
	return {'B03': (offset,10000.0), 'B08': (offset,10000.0)}


def compute_ndwi(g_dn,n_dn,rad):
	'''
	NDWI from raw L2A DNs. Reflectance below zero (possible after the offset,
	e.g. dark water in NIR) is clipped to zero; where both bands are zero the
	ratio is undefined and NDWI is set to 0. Returns (ndwi, n_undefined).
	'''
	g = np.clip((g_dn.astype(np.float32) + rad['B03'][0]) / rad['B03'][1],0,None)
	n = np.clip((n_dn.astype(np.float32) + rad['B08'][0]) / rad['B08'][1],0,None)
	den  = g + n
	ndwi = np.divide(g-n,den,out=np.zeros_like(den),where=den > 0)
	return ndwi, int((den <= 0).sum())

################################################################################
# THRESHOLDS + METRICS
################################################################################
def to_bins(ndwi):
	'''
	Bin index k such that NDWI >= EDGES[k] <=> bin >= k.
	'''
	return np.clip(((ndwi + 1.0) * 0.5 * N_BINS).astype(np.int64),0,N_BINS-1)


def otsu_bin(hist):
	'''
	Otsu split on a histogram: returns k so that bins >= k are the upper class.
	'''
	p   = hist.astype(np.float64)
	w0  = np.cumsum(p)[:-1]            # WEIGHT OF BINS < k, FOR k = 1..N-1
	w1  = p.sum() - w0
	mid = np.arange(N_BINS) + 0.5
	m0  = np.cumsum(p*mid)[:-1]
	m1  = (p*mid).sum() - m0
	ok  = (w0 > 0) & (w1 > 0)
	between = np.zeros_like(w0)
	between[ok] = w0[ok]*w1[ok] * (m0[ok]/w0[ok] - m1[ok]/w1[ok])**2
	return int(np.argmax(between)) + 1


def confusion_at(hist_w,hist_l,k):
	'''
	(TP,FP,FN,TN) for the rule NDWI >= EDGES[k], from water/land histograms.
	'''
	tp = int(hist_w[k:].sum()); fn = int(hist_w[:k].sum())
	fp = int(hist_l[k:].sum()); tn = int(hist_l[:k].sum())
	return tp,fp,fn,tn


def best_bin(hist_w,hist_l):
	'''
	Threshold bin maximising water IoU = TP/(TP+FP+FN).
	'''
	tp = np.cumsum(hist_w[::-1])[::-1].astype(np.float64)  # TP for NDWI >= EDGES[k]
	fp = np.cumsum(hist_l[::-1])[::-1].astype(np.float64)
	fn = hist_w.sum() - tp
	iou = tp / np.maximum(tp+fp+fn,1)
	return int(np.argmax(iou))


def metrics(tp,fp,fn,tn):
	'''
	Same definitions as calculate_metrics() in models/train.py.
	'''
	d = lambda a,b: a/b if b > 0 else 0.0
	return {
		'iou1': d(tp,tp+fp+fn), 'iou0': d(tn,tn+fn+fp),
		'ppv1': d(tp,tp+fp),    'tpr1': d(tp,tp+fn),
		'acc':  d(tp+tn,tp+fp+fn+tn)}

################################################################################
# PER-PRODUCT WORKER
################################################################################
def window_resid(chip_band,raw_dn):
	'''
	Alignment check. chip.py wrote each chip band as round(DN/p99*255), so on
	pixels the 99th-percentile clip left untouched (chip < 255) the chip is an
	exact linear function of the raw DN up to rounding. Fit chip ~ a*DN + b and
	return the largest absolute residual: ~0.5 for the right window, several
	levels for a window shifted by even one pixel (every chip has a shoreline).
	'''
	keep = chip_band < 255
	if keep.sum() < 100:
		return float('nan')
	y = chip_band[keep].astype(np.float64); x = raw_dn[keep].astype(np.float64)
	if x.std() == 0:
		return float('nan')
	a,b = np.polyfit(x,y,1)
	return float(np.abs(y - (a*x + b)).max())


def process_product(base_id,safe_path,dw_path,chips,split_dir):
	'''
	Score every chip of one S2/DW product. chips: [(chip_name,row,col)].
	Returns pooled water/land NDWI histograms, pooled per-chip-Otsu confusion,
	and one record per chip.
	'''
	rad     = l2a_radiometry(safe_path)
	product = Product(safe_path,dw_path)
	try:
		windows = dict((tuple(ij),w) for ij,w in get_strided_windows(product.s2_borders))
		by_band = {os.path.basename(r.files[0]).split('_')[-2]: r for r in product.s2_readers}
		g_rdr,n_rdr = by_band['B03'],by_band['B08']

		hist_w = np.zeros(N_BINS,np.int64)
		hist_l = np.zeros(N_BINS,np.int64)
		otsu_conf = np.zeros(4,np.int64)
		records = []

		for name,row,col in chips:
			w     = windows[(row,col)]
			g_dn  = g_rdr.read(1,window=w)
			n_dn  = n_rdr.read(1,window=w)
			bands = np.array(Image.open(f"{split_dir}/{name}_B0X.tif"))   # H,W,4 = R,G,B,NIR
			water = np.array(Image.open(f"{split_dir}/{name}_LBL.tif")) == 255

			# ALIGNMENT CHECK -- BOTH BANDS
			resid = max(window_resid(bands[...,1],g_dn),window_resid(bands[...,3],n_dn))
			rec   = {'chip': name, 'resid': resid, 'water_frac': float(water.mean())}
			if not resid <= MAX_RESID: # ALSO CATCHES NaN
				rec['status'] = 'misaligned'
				records.append(rec)
				continue

			ndwi,n_undef = compute_ndwi(g_dn,n_dn,rad)
			b = to_bins(ndwi)
			hw = np.bincount(b[water],minlength=N_BINS)
			hl = np.bincount(b[~water],minlength=N_BINS)
			hist_w += hw; hist_l += hl

			k = otsu_bin(hw + hl) # LABEL-FREE: HISTOGRAM OF ALL PIXELS
			conf = confusion_at(hw,hl,k)
			otsu_conf += conf

			rec.update({'status': 'ok', 'n_undefined': n_undef,
				'otsu_t': float(EDGES[k]),
				'iou1_ndwi0': metrics(*confusion_at(hw,hl,N_BINS//2))['iou1'],
				'iou1_otsu':  metrics(*conf)['iou1']})
			records.append(rec)
	finally:
		product.close()

	return {'hist_w': hist_w, 'hist_l': hist_l, 'otsu_conf': otsu_conf, 'records': records}

################################################################################
# DISCOVERY
################################################################################
def parse_chip(fname):
	'''
	{date}_{datastrip}_{tile}_{orbit}_{row}_{col}_B0X.tif -> (name, base_id, row, col)
	'''
	name = fname[:-8]
	return name, name[:-6], int(name[-5:-3]), int(name[-2:])


def index_products(s2_dir,dw_dir):
	'''
	base_id -> (safe_path, dw_path), built exactly as chip.py names products.
	'''
	index = {}
	for rel in glob.glob(SAFE_GLOB,root_dir=s2_dir):
		safe  = f"{s2_dir}/{rel}"
		dw_id = get_dw_id(safe,get_datastrip_id(safe))
		orbit = os.path.basename(safe).split('_')[4]
		index[f"{dw_id[:-4]}_{orbit}"] = (safe,f"{dw_dir}/{dw_id}")
	return index


def run_split(split,args,index):
	'''
	Process all chips of one split. Returns pooled histograms, Otsu confusion,
	chip records, and the number of chips that could not be evaluated.
	'''
	split_dir = f"{args.chip_dir}/{split}"
	chips = {}
	for f in sorted(glob.glob("*_B0X.tif",root_dir=split_dir)):
		name,base_id,row,col = parse_chip(f)
		chips.setdefault(base_id,[]).append((name,row,col))
	n_chips = sum(len(v) for v in chips.values())

	missing = [b for b in chips if b not in index or not os.path.isfile(index[b][1])]
	for b in missing:
		print(f"  [{split}] no S2/DW product for {b} -- skipping {len(chips[b])} chip(s)",file=sys.stderr)

	todo = [b for b in chips if b not in missing]
	print(f"[{split}] {n_chips} chips in {len(chips)} products; evaluating {len(todo)} products",flush=True)

	hist_w = np.zeros(N_BINS,np.int64)
	hist_l = np.zeros(N_BINS,np.int64)
	otsu_conf = np.zeros(4,np.int64)
	records = []
	with ProcessPoolExecutor(max_workers=args.workers) as ex:
		futures = {ex.submit(process_product,b,*index[b],chips[b],split_dir): b for b in todo}
		for i,fut in enumerate(as_completed(futures)):
			out = fut.result()
			hist_w += out['hist_w']; hist_l += out['hist_l']; otsu_conf += out['otsu_conf']
			for r in out['records']:
				r['split'] = split
			records += out['records']
			if (i+1) % 10 == 0 or (i+1) == len(todo):
				print(f"  [{split}] {i+1}/{len(todo)} products",flush=True)

	n_skipped = sum(len(chips[b]) for b in missing) + sum(r['status'] != 'ok' for r in records)
	return {'hist_w': hist_w, 'hist_l': hist_l, 'otsu_conf': otsu_conf,
		'records': records, 'n_chips': n_chips, 'n_skipped': n_skipped}

################################################################################
# MAIN
################################################################################
def parse_args():
	parser = argparse.ArgumentParser(description="NDWI threshold baseline from L2A reflectance.")
	parser.add_argument('--s2-dir',required=True,help="Directory holding eodata/Sentinel-2/MSI/L2A/...")
	parser.add_argument('--dw-dir',required=True,help="Directory of DynamicWorld label .tif files.")
	parser.add_argument('--chip-dir',required=True,help="Split chip directory (<chips>_sorted).")
	parser.add_argument('--eval-split',default='testing',help="Split to report on (default: testing).")
	parser.add_argument('--fit-split',default='validation',help="Split the global threshold is fit on (default: validation).")
	parser.add_argument('--out-dir',default='./ndwi_baseline',help="Output directory for the .tsv files.")
	parser.add_argument('--workers',type=int,default=4,help="Products processed in parallel.")
	args = parser.parse_args()
	for d in (args.s2_dir,args.dw_dir,args.chip_dir):
		assert os.path.isdir(d), f"Directory not found: {d}"
	for s in {args.eval_split,args.fit_split}:
		assert os.path.isdir(f"{args.chip_dir}/{s}"), f"Split not found: {args.chip_dir}/{s}"
	args.s2_dir   = args.s2_dir.rstrip('/')
	args.dw_dir   = args.dw_dir.rstrip('/')
	args.chip_dir = args.chip_dir.rstrip('/')
	return args


if __name__ == '__main__':
	args = parse_args()
	os.makedirs(args.out_dir,exist_ok=True)

	index   = index_products(args.s2_dir,args.dw_dir)
	splits  = list(dict.fromkeys([args.fit_split,args.eval_split])) # FIT FIRST, NO DUPLICATE
	results = {s: run_split(s,args,index) for s in splits}

	# GLOBAL THRESHOLD FIT ON --fit-split ONLY
	fit = results[args.fit_split]
	if fit['n_chips'] == fit['n_skipped']:
		sys.exit(f"error: no chip of '{args.fit_split}' could be evaluated "
			f"(missing products or failed alignment) -- nothing to fit a threshold on")
	k_fit = best_bin(fit['hist_w'],fit['hist_l'])

	rows = []
	for s in splits:
		r = results[s]
		k_oracle = best_bin(r['hist_w'],r['hist_l'])
		methods = [
			('ndwi>=0',              EDGES[N_BINS//2], confusion_at(r['hist_w'],r['hist_l'],N_BINS//2)),
			(f'fit:{args.fit_split}', EDGES[k_fit],     confusion_at(r['hist_w'],r['hist_l'],k_fit)),
			('otsu-per-chip',         float('nan'),     tuple(r['otsu_conf'])),
			('oracle',                EDGES[k_oracle],  confusion_at(r['hist_w'],r['hist_l'],k_oracle)),
		]
		for method,t,conf in methods:
			rows.append({'split': s, 'method': method, 'threshold': t,
				'chips': r['n_chips'] - r['n_skipped'], 'skipped': r['n_skipped'], **metrics(*conf)})

	# SUMMARY
	cols = ['split','method','threshold','chips','skipped','iou1','iou0','ppv1','tpr1','acc']
	with open(f"{args.out_dir}/ndwi_summary.tsv",'w') as fp:
		fp.write('\t'.join(cols)+'\n')
		for r in rows:
			fp.write('\t'.join(f"{r[c]:.5f}" if isinstance(r[c],float) else str(r[c]) for c in cols)+'\n')

	# PER CHIP
	ccols = ['split','chip','status','resid','water_frac','n_undefined','otsu_t','iou1_ndwi0','iou1_otsu']
	with open(f"{args.out_dir}/ndwi_chips.tsv",'w') as fp:
		fp.write('\t'.join(ccols)+'\n')
		for s in splits:
			for r in sorted(results[s]['records'],key=lambda r: r['chip']):
				fp.write('\t'.join(f"{r[c]:.5f}" if isinstance(r.get(c),float) else str(r.get(c,'')) for c in ccols)+'\n')

	# STDOUT
	print()
	print(f"{'split':<11}{'method':<17}{'t':>8}{'chips':>7}{'skip':>6}{'IoU_1':>9}{'IoU_0':>9}{'PPV_1':>9}{'TPR_1':>9}")
	for r in rows:
		t = '-' if np.isnan(r['threshold']) else f"{r['threshold']:.3f}"
		print(f"{r['split']:<11}{r['method']:<17}{t:>8}{r['chips']:>7}{r['skipped']:>6}"
			f"{r['iou1']:>9.4f}{r['iou0']:>9.4f}{r['ppv1']:>9.4f}{r['tpr1']:>9.4f}")
	misaligned = sum(rec['status'] == 'misaligned' for s in splits for rec in results[s]['records'])
	if misaligned:
		print(f"\nWARNING: {misaligned} chip(s) failed the alignment check (residual > {MAX_RESID}); "
			f"see status in {args.out_dir}/ndwi_chips.tsv",file=sys.stderr)
	print(f"\nWritten: {args.out_dir}/ndwi_summary.tsv, {args.out_dir}/ndwi_chips.tsv")
