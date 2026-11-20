import torch
import torchvision
from torchvision import tv_tensors
from PIL import Image
import torchvision.transforms as v1
import torchvision.transforms.v2 as v2
import glob
import os
import tifffile as tiff
import numpy as np
import time
import argparse

MEAN = torch.tensor([123.305154,132.731427,131.599954,115.507302]).view(-1,1,1)
STDD = torch.tensor([53.223368,51.529520,53.707588,55.610260]).view(-1,1,1)

################################################################################
# FUNCTIONS/HELPERS
################################################################################
def build_cached_dataset(chip_dir,cache_dir,split):

	# splits = ("training","validation","testing")

	band_files = sorted(glob.glob(f"{chip_dir}/{split}/*_B0X.tif"))
	chips = [os.path.basename(i)[0:-8] for i in band_files] # CHIP NAMES ONLY, glob RETURNS FULL PATHS
	n = len(chips)

	# BUFFER
	vnir = np.lib.format.open_memmap(f"{cache_dir}/{split}_bands.npy",mode='w+',dtype=np.uint8,shape=(n,4,256,256))
	mask = np.lib.format.open_memmap(f"{cache_dir}/{split}_masks.npy",mode='w+',dtype=np.uint8,shape=(n,256,256))

	for i,c in enumerate(chips):
		vnir[i] = tiff.imread(f"{chip_dir}/{split}/{c}_B0X.tif").transpose(2,0,1) # HWC -> CHW
		mask[i] = tiff.imread(f"{chip_dir}/{split}/{c}_LBL.tif")

	for arr in (vnir,mask):
		arr.flush()

	print(f"Cached bands written to {cache_dir}/{split}_bands.npy")
	print(f"Cached labels written to {cache_dir}/{split}_masks.npy")

	with open(f"{cache_dir}/{split}_ids.txt","w") as fp:
		fp.write('\n'.join(chips))


def load_cached_split(cache_dir,split):
	print(f"Loading cached .npy files in {cache_dir}...")
	t0_disk_to_ram = time.perf_counter()
	vnir = np.load(f"{cache_dir}/{split}_bands.npy")
	mask = np.load(f"{cache_dir}/{split}_masks.npy")
	t1_disk_to_ram = time.perf_counter()
	print("Loaded to RAM! -- ",end='')
	print(f"{t1_disk_to_ram-t0_disk_to_ram:.2f} secs.")
	return vnir,mask


S1F11_IGNORE = 127       # SEN1FLOODS11 MASK VALUE FOR PIXELS TO IGNORE (LAND 0, WATER 255 AS IN S2DW)
S1F11_RGBN   = [3,2,1,7] # S2Hand BAND INDICES OF B04,B03,B02,B08 -- S2DW ORDER R,G,B,NIR
S1F11_CSV    = {
	'train':   'flood_train_data.csv',
	'valid':   'flood_valid_data.csv',
	'test':    'flood_test_data.csv',
	'bolivia': 'flood_bolivia_data.csv'}

def build_cached_s1f11(s1f11_dir,cache_dir,split,dn_offset=1000,tile=256):
	'''
	Cache a hand-labeled Sen1Floods11 split in the S2DW .npy format. Bands are
	prepared as in eval_sen1floods11.py: B04,B03,B02,B08 -> R,G,B,NIR; +dn_offset
	(S2DW stretched 2025 L2A DNs, which carry a +1000 offset); clip at the per-chip
	99th percentile of valid pixels and scale to 0-255; invalid pixels 0.
	Each 512x512 chip becomes (512/tile)^2 tiles, row-major. Writes:
		{cache_dir}/s1f11_{split}_bands.npy  uint8 [N,4,tile,tile]
		{cache_dir}/s1f11_{split}_masks.npy  uint8 [N,tile,tile]  0 land, 255 water,
		                                     S1F11_IGNORE where LabelHand is -1 or a band is 0
		{cache_dir}/s1f11_{split}_ids.txt    one tile per line: <chip id>_<row>_<col>
	split is one of train, valid, test, bolivia. Chips with < 100 valid pixels are
	kept as all-ignore tiles so the ids stay complete. Load with
	load_cached_split(cache_dir,f"s1f11_{split}").
	'''
	root = s1f11_dir.rstrip('/')
	if os.path.isdir(f"{root}/v1.1"):
		root = f"{root}/v1.1"
	assert os.path.isdir(f"{root}/splits/flood_handlabeled"), f"No splits/flood_handlabeled under {root}"
	base = f"{root}/data/flood_events/HandLabeled"

	# CHIP IDS (e.g. Ghana_313799): CSV ROWS ARE "<id>_S1Hand.tif,<id>_LabelHand.tif" WITH CRLF ENDINGS
	with open(f"{root}/splits/flood_handlabeled/{S1F11_CSV[split]}",'r') as fp:
		chips = [l.strip().split(',')[0].replace('_S1Hand.tif','') for l in fp if l.strip()]

	CHIP = 512
	assert CHIP % tile == 0, f"tile {tile} does not divide {CHIP}"
	k    = CHIP // tile
	n    = len(chips) * k * k

	pfx  = f"{cache_dir}/s1f11_{split}"
	vnir = np.lib.format.open_memmap(f"{pfx}_bands.npy",mode='w+',dtype=np.uint8,shape=(n,4,tile,tile))
	mask = np.lib.format.open_memmap(f"{pfx}_masks.npy",mode='w+',dtype=np.uint8,shape=(n,tile,tile))

	ids,n_empty,n_px = [],0,np.zeros(3,np.int64) # n_px: LAND, WATER, IGNORED
	for i,cid in enumerate(chips):
		s2  = tiff.imread(f"{base}/S2Hand/{cid}_S2Hand.tif")                   # int16, 13 BANDS
		if s2.shape[0] != 13: # PIXEL-INTERLEAVED FILE -> BAND FIRST
			s2 = np.moveaxis(s2,-1,0)
		lbl = np.squeeze(tiff.imread(f"{base}/LabelHand/{cid}_LabelHand.tif")) # -1/0/1
		assert s2.shape[1:] == lbl.shape == (CHIP,CHIP), f"{cid}: unexpected shape {s2.shape} / {lbl.shape}"

		valid = (lbl != -1) & np.all(s2[S1F11_RGBN] > 0,axis=0)
		img   = np.zeros((4,CHIP,CHIP),np.uint8)
		if valid.sum() < 100: # NOTHING TO STRETCH OR LEARN FROM
			valid[:] = False
			n_empty += 1
		else:
			for ch,b in enumerate(S1F11_RGBN): # PER-BAND 99th-PERCENTILE STRETCH -> 0-255
				x  = s2[b].astype(np.float64) + dn_offset
				hi = int(np.percentile(x[valid],99))
				if hi > 0:
					img[ch] = np.round(np.clip(x,0,hi) / hi * 255).astype(np.uint8)
			img[:,~valid] = 0
		lbl8 = np.where(valid,np.where(lbl == 1,255,0),S1F11_IGNORE).astype(np.uint8)
		n_px += [(lbl8 == 0).sum(),(lbl8 == 255).sum(),(~valid).sum()]

		for r in range(k):
			for c in range(k):
				j = (i*k + r)*k + c
				vnir[j] = img[:,r*tile:(r+1)*tile,c*tile:(c+1)*tile]
				mask[j] = lbl8[r*tile:(r+1)*tile,c*tile:(c+1)*tile]
				ids.append(f"{cid}_{r}_{c}")

	for arr in (vnir,mask):
		arr.flush()
	with open(f"{pfx}_ids.txt","w") as fp:
		fp.write('\n'.join(ids))

	land,water,ign = n_px
	print(f"[s1f11 {split}] {len(chips)} chips -> {n} tiles ({n_empty} chips with no valid pixels) | "
		f"pixels: land {land}, water {water} ({water/max(land+water,1):.1%} of valid), ignored {ign}")
	print(f"Cached bands written to {pfx}_bands.npy")
	print(f"Cached labels written to {pfx}_masks.npy")

################################################################################
# CLASSES
################################################################################
class TrainTransform:
	'''
	Class for the augmentation performed during training. Image and label use
	same geometric rotation/flip. Color jitter and blur applied only to bands.
	'''
	def __init__(self):
		self.geometric = v2.Compose([
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5),
			v2.RandomChoice([
				v2.RandomRotation([0,0]),
				v2.RandomRotation([90,90]),
				v2.RandomRotation([180,180]),
				v2.RandomRotation([270,270])
			])
		])

		self.intensity = v2.Compose([
			v2.GaussianNoise(mean=0,sigma=0.02*255,clip=False)
		])


	def __call__(self,image,label):
		image, label = self.geometric(image,label)
		image        = self.intensity(image)
		return image, label


class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3,n_labels=2,transform=None):

		# GET LIST OF CHIPS/STRINGS
		self.dir        = chip_dir
		self.band_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
		self.ids        = [i[0:-8] for i in self.band_files]

		# NORMALIZING CONSTANTS
		self.mean = torch.tensor([123.305154,132.731427,131.599954,115.507302]).view(-1,1,1)
		self.std  = torch.tensor([53.223368,51.529520,53.707588,55.610260]).view(-1,1,1)

		# ADJUST ARRAY GEOMETRIES ACCORDING TO NR OF BANDS
		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")
		if n_bands == 3:
			self.band_func = self.get_rgb
			self.mean = self.mean[0:3]
			self.std  = self.std[0:3]
		if n_bands == 4:
			self.band_func = self.get_vnir

		# ADJUST LABEL INDICES ACCORDING TO NR OF LABELS
		if (n_labels)!=3 and (n_labels!=2):
			raise ValueError("Incorrect number of target labels.")
		if n_labels == 2:
			self.lbl_div = 255
		if n_labels == 3:
			self.lbl_div = 127

		# TRANSFORMS ONLY FOR TRAIN SET -- AUGMENTATION
		self.train_transform = transform


	def get_rgb(self,idx):
		'''
		Return 3 bands.
		'''
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()	
		return Image.merge(mode='RGB',bands=[r,g,b])


	def get_vnir(self,idx):
		'''
		Return 4 bands.
		'''
		return Image.open(f'{self.ids[idx]}_B0X.tif')


	def load_image(self,idx):
		'''
		Return as float32 tensor wrapped in tv_tensor.Image class.
		Values 0-255
		'''
		img = v2.functional.pil_to_tensor(self.band_func(idx))
		img = img.to(torch.float32)
		return tv_tensors.Image(img)


	def load_label(self,idx):
		'''
		Adjust to discrete labels and return as tv_tensors.Mask class.
		Values 0,1
		'''
		lbl = v2.functional.pil_to_tensor(Image.open(f'{self.ids[idx]}_LBL.tif'))
		lbl = torch.squeeze(lbl,0)
		lbl = torch.div(lbl,self.lbl_div,rounding_mode='floor').to(torch.int64)
		return tv_tensors.Mask(lbl)


	def __len__(self):
		return len(self.ids)


	def __getitem__(self,idx):

		# GET RGB (or RGB+NIR) & LABELS
		image = self.load_image(idx) #tv_tensors.Image object
		label = self.load_label(idx) #tv_tensors.Mask object

		# AUGMENT -- TRAINING
		if self.train_transform:
			image,label = self.train_transform(image,label)

		# NORMALIZE & RETURN
		image = (image - self.mean) / self.std
		return image,label


################################################################################
# FAST DATASETS (Loaded in RAM & using uint8 only)
################################################################################
class SentinelFastDataset(torch.utils.data.Dataset):
	def __init__(self,bands_ndarray,masks_ndarray,n_bands=4):
		self.bands = torch.from_numpy(bands_ndarray)
		self.masks = torch.from_numpy(masks_ndarray)
		self.n       = self.bands.shape[0]
		self.n_bands = n_bands

	def __len__(self):
		return self.n

	def __getitem__(self,idx):
		vnir = self.bands[idx,0:self.n_bands]
		mask = self.masks[idx]
		return vnir,mask


class GPUTransform(torch.nn.Module):
	'''
	On-GPU training transform. Takes the uint8 image [B,C,H,W] and uint8 mask [B,1,H,W] (0/255) 
	already on the device (square chips, H == W).
	Applies the same random flip/transpose to image and mask, then gain to the image only.
	Returns the normalized float32 image and the int64 class mask [B,H,W].

	Use:
		gpu_tf = dataloader.GPUTransform().to(device)
		for x,y in loader:                    # both uint8
			x,y = gpu_tf(x.to(device,non_blocking=True),y.to(device,non_blocking=True))
	'''
	def __init__(self,gain=0.10):
		super().__init__()
		self.gain = gain
		self.register_buffer('mean',MEAN.clone())
		self.register_buffer('stdd',STDD.clone())

	def dihedral(self,x):
		# PER-SAMPLE FLIPS x TRANSPOSE = THE 8 DIHEDRAL ELEMENTS
		flip_v,flip_h,transp = (torch.rand(3,x.shape[0],1,1,1,device=x.device) < 0.5)
		x = torch.where(flip_v,x.flip(2),x)
		x = torch.where(flip_h,x.flip(3),x)
		return torch.where(transp,x.transpose(2,3),x)

	@torch.no_grad()
	def forward(self,x,y):
		# MASK RIDES ALONG AS AN EXTRA CHANNEL SO BOTH GET THE SAME GEOMETRY
		y   = y.view(x.shape[0],1,*x.shape[-2:]) # [B,H,W] OR [B,1,H,W] -> [B,1,H,W]
		xy  = self.dihedral(torch.cat([x,y],dim=1))
		x,y = xy[:,:-1].float(),xy[:,-1] # [:,-1] DROPS THE CHANNEL DIM -> [B,H,W]

		# GAIN SCALING ON UNIFORM DIST [1-gain,1+gain]
		if self.gain > 0:
			g = self.gain
			x = x * torch.empty(x.shape[0],1,1,1,device=x.device).uniform_(1-g,1+g)

		# NORMALIZE IMAGE, 0/255 MASK -> 0/1 CLASS INDICES
		x = (x-self.mean)/self.stdd
		y = y.div(255,rounding_mode='floor').long()

		return x,y


class GPUValidationTransform(torch.nn.Module):
	'''
	GPU band normalization. No augmentation (only cast + normalize).
	Takes the uint8 batch [B,4,H,W] and uint8 mask [B,1,H,W] (0/255).
	Returns the float32 image and the int64 class mask [B,H,W].
	'''
	def __init__(self):
		super().__init__()
		self.register_buffer('mean',MEAN.clone())
		self.register_buffer('stdd',STDD.clone())

	@torch.no_grad()
	def forward(self,x,y):
		x = (x.float()-self.mean)/self.stdd
		y = y.squeeze(1).div(255,rounding_mode='floor').long() # 0/255 MASK -> 0/1 CLASS INDICES
		return x,y


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir',required=True,help='Directory of chip dataset')
	parser.add_argument('--cache-dir',required=True,help='Directory of output .npy cached datasets')
	args = parser.parse_args()
	args.data_dir = args.data_dir.rstrip('/')
	args.cache_dir = args.cache_dir.rstrip('/')
	splits = ("training","validation","testing")
	for s in splits:
		build_cached_dataset(args.data_dir,args.cache_dir,s)


class S1F11FastDataset(torch.utils.data.Dataset):

	def __init__(self,bands_ndarray,masks_ndarray):
		pass

	def __len__(self):
		pass

	def __getitem__(self,idx):
		pass

