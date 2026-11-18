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