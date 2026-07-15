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


class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3,n_labels=2,transform=None):
		self.dir        = chip_dir
		self.vnir_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
		self.ids        = [i[0:-8] for i in self.vnir_files]

		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")
		if n_bands == 3:
			self.input_func = self.rgb_get
		if n_bands == 4:
			self.input_func = self.vnir_get

		if (n_labels)!=3 and (n_labels!=2):
			raise ValueError("Incorrect number of target labels.")
		if n_labels == 2:
			lbl_div = 255
		if n_labels == 3:
			lbl_div = 127

		self.input_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=True)		
		])

		self.label_transform = v2.Compose([
			v2.ToImage(),
			# v2.Lambda(lambda x: torch.squeeze(x,0)),
			# v2.Lambda(lambda x: torch.div(x,lbl_div,rounding_mode='floor')),
			LabelDivTransform(lbl_div=lbl_div), #Removed lambda for ddp func pickling
			v2.ToDtype(torch.int64)
		])

		self.additional_transform = transform #applied only to train, not valdtn/test

	def rgb_get(self,idx):
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()		
		return Image.merge(mode='RGB',bands=[r,g,b])

	def vnir_get(self,idx):
		return Image.open(f'{self.ids[idx]}_B0X.tif')

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		image = self.input_transform(self.input_func(idx))
		label = self.label_transform(Image.open(f'{self.ids[idx]}_LBL.tif'))
		if self.additional_transform:
			image,label = self.additional_transform(image,label)
		return image,label

