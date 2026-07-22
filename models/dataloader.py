import torch
import torchvision as tv
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import glob



################################################################################
# CLASSES
################################################################################
class LabelDivTransform(torch.nn.Module):
	'''
	Labels in the S2-DW dataset are stored as 255 and 0 for two classes and 255,
	127, and 0 for 3 classes. This is for ease of visualization/inspection.
	Labels are converted to values in [1,0] or [2,1,0] by floor division. In 
	either case water is 1 and land is 0.
	'''
	def __init__(self,lbl_div):
		super().__init__()
		self.lbl_div = lbl_div


	def forward(self,lbl):
		'''
		Parameters
		----------
		lbl : torch.Tensor
		    The label array. It's ingested with dimension [1,H,W,L], i.e.: height, 
		    width, and L number of 2 or 3 classes.
		Returns
		-------
		torch.Tensor
		    Converted label with values 0 and 1 for the binary case; and 0,1, and 2
		    for three class arrays.	
		'''		
		lbl = torch.squeeze(lbl,0)
		lbl = torch.div(lbl,self.lbl_div,rounding_mode='floor')
		return lbl



class TrainTransform:
	'''
	Class for the augmentation performed during training. Image and label use
	same geometric rotation/flip. Color jitter and blur applied only to bands.
	'''
	def __init__():
		self.geometric = v2.Compose([
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5)
		])

		self.intensity = v2.Compose([
			v2.ColorJitter(brightness=0.2,contrast=0.2),
			v2.GaussianNoise(sigma=0.02)
		])


	def __call__(self,image,label):
		image, label = self.geometric(image,label)
		image        = self.intensity(image)
		return image, label



class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3,n_labels=2,transform=None):

		# GET LIST OF CHIPS
		self.dir        = chip_dir
		self.vnir_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
		self.ids        = [i[0:-8] for i in self.vnir_files]

		# NORMALIZING VARIABLES
		self.mean = torch.tensor([123.30515418,132.73142713,131.5999535,115.50730149]).view(-1,1,1)
		self.std  = torch.tensor([53.223368,51.52951993,53.70758823,55.61025949]).view(-1,1,1)

		# ADJUST ARRAY GEOMETRIES ACCORDING TO NR OF BANDS
		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")
		if n_bands == 3:
			self.input_func = self.rgb_get
			self.mean = self.mean[0:3]
			self.std  = self.std[0:3]
		if n_bands == 4:
			self.input_func = self.vnir_get

		# ADJUST LABEL INDICES ACCORDING TO NR OF LABELS
		if (n_labels)!=3 and (n_labels!=2):
			raise ValueError("Incorrect number of target labels.")
		if n_labels == 2:
			lbl_div = 255
		if n_labels == 3:
			lbl_div = 127

		# REQUIRED TRANSFORMS FOR ALL INPUTS
		self.input_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=False)		
		])

		# REQUIRED TRANSFORMS FOR ALL LABELS
		self.label_transform = v2.Compose([
			v2.ToImage(),
			LabelDivTransform(lbl_div=lbl_div), #ddp pickling not working w/ lambdas
			v2.ToDtype(torch.int64)
		])

		# TRANSFORMS ONLY FOR TRAIN SET -- AUGMENTATION
		self.train_transform = transform


	def rgb_get(self,idx):
		'''
		Return 3 bands.
		'''
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()		
		return Image.merge(mode='RGB',bands=[r,g,b])


	def vnir_get(self,idx):
		'''
		Return 4 bands.
		'''
		return Image.open(f'{self.ids[idx]}_B0X.tif')


	def __len__(self):
		return len(self.ids)


	def __getitem__(self,idx):

		# GET RGB (or RGB-NIR) & LABELS
		image = self.input_transform(self.input_func(idx))
		label = self.label_transform(Image.open(f'{self.ids[idx]}_LBL.tif'))

		# AUGMENT -- TRAINING
		if self.train_transform:
			image,label = self.train_transform(image,label)

		# NORMALIZE
		image = (image - self.mean) / self.std

		return image,label

