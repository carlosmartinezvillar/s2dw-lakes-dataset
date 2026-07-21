import torch
import torch.nn as nn
# import torch.nn.functional as F
import math

################################################################################
# CNN Blocks
################################################################################
class ConvBlock(nn.Module):
	'''
	Base convolutional block in stage of hierarchy.
	Channel dimension consistent throughout block to match skip/residual.
	'''
	def __init__(self,channels,depth=2):
		super().__init__()
		self.block = nn.ModuleList()
		# self.block = nn.Sequential(
			# nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
			# nn.GroupNorm(1,channels),
			# nn.GELU(),
			# nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
			# nn.GroupNorm(1,channels),
			# nn.GELU()
		# )
		for i in range(depth):
			self.block.append(nn.Sequential(
				nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
				nn.GroupNorm(1,channels),
				nn.GELU()
			))

	def forward(self,x):
		for layer in self.block:
			out = layer(x)
		return x + out


class ConvBlockSeparable(nn.Module):
	'''
	Base convolutional block, with separable convolutions.
	Channel dimension consistent throughout block to match skip/residual.
	'''
	def __init__(self,channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(channels,channels,3,1,padding=1,groups=channels,bias=True),
			nn.Conv2d(channels,channels,1,1,padding=0,bias=True),
			nn.GroupNorm(1,channels),
			nn.GELU(),
			nn.Conv2d(channels,channels,3,1,padding=1,groups=channels,bias=True),
			nn.Conv2d(channels,channels,1,1,padding=0,bias=True),
			nn.GroupNorm(1,channels),
			nn.GELU()
		)

	def forward(self,x):
		return x + self.block(x)	

################################################################################
# ViT Blocks
################################################################################
class MultiHeadSelfAttention(nn.Module):
	'''
	Multi-head Self-Attention Operation
	B: batch dimension
	E: embedding dimensino
	N: sequence length
	H: head dimension
	'''
	def __init__(self, E, num_heads=4):
		super().__init__()
		assert E % num_heads == 0, f"channels={E} not divisible by num_heads={num_heads}"
		self.E         = E
		self.num_heads = num_heads
		self.head_dim  = E // num_heads
		self.W_qkv  = nn.Linear(E, E * 3, bias=False)
		self.W_o    = nn.Linear(E, E, bias=False)

	def forward(self, x):
		B, N, _ = x.shape
		QKV = self.W_qkv(x)         # [B,N,3E]
		Q,K,V = QKV.chunk(3,dim=-1) # each [B,N,E]
		Q = Q.view(B,N,self.num_heads,self.head_dim).transpose(1,2) # [B,num_heads,N,H]
		K = K.view(B,N,self.num_heads,self.head_dim).transpose(1,2)
		V = V.view(B,N,self.num_heads,self.head_dim).transpose(1,2)

		attn = (Q @ K.transpose(-2, -1)) # [B,num_heads,N,N]
		attn = attn / (self.head_dim ** 0.5)
		attn = attn.softmax(dim=-1)

		x = attn @ V # [B,num_heads,N,H]
		x = x.transpose(1, 2).reshape(B,N,self.E) #[B,num_heads,N,H] -> [B,N,num_heads,H] -> [B,N,E]
		return self.W_o(x) # [B,N,E]


class MultiHeadCrossAttention(nn.Module): #<<< option?
	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		assert d_model % num_heads == 0
		self.num_heads = num_heads
		self.head_dim  = d_model // num_heads
		self.W_q       = nn.Linear(d_model, d_model, bias=False)
		self.W_kv      = nn.Linear(d_model, 2 * d_model, bias=False)
		self.W_out     = nn.Linear(d_model, d_model, bias=False)
		self.dropout   = nn.Dropout(dropout)

	def forward(self, x, enc_out):
		'''
		query:     x
		key,value: enc_out
		'''
		B, N, _ = x.shape
		S        = enc_out.size(1)
		Q        = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		K, V     = self.W_kv(enc_out).chunk(2, dim=-1)
		K        = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
		V        = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)


		attn = (Q @ K.transpose(-2, -1)) # [B,num_heads,N,N]
		attn = attn / (self.head_dim ** 0.5)
		attn = attn.softmax(dim=-1)

		x = attn @ V # [B,num_heads,N,H]
		x = x.transpose(1, 2).contiguous().view(B, N, -1)
		return self.W_out(x)


class MLP(nn.Module):
	'''
	Vanilla MLP layer in transformer block
	'''
	def __init__(self, dim, mlp_ratio=4):
		super().__init__()
		hidden_dim = dim * mlp_ratio
		self.layers = nn.Sequential(
		    nn.Linear(dim, hidden_dim),
		    nn.GELU(),
		    nn.Linear(hidden_dim, dim)
		)

	def forward(self, x):
		return self.layers(x)


class ViTLayer(nn.Module):
	'''
	A complete ViT layer (i.e. MHSA + MLP)
	'''
	def __init__(self,E,num_heads,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.norm2 = nn.LayerNorm(E)
		self.mlp   = MLP(E, mlp_ratio)

	def forward(self, tokens):
		tokens = tokens + self.attn(self.norm1(tokens))
		tokens = tokens + self.mlp(self.norm2(tokens))
		return tokens


class ViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	'''
	def __init__(self,E,num_heads,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ViTLayer(E,num_heads,mlp_ratio) for _ in range(depth)])
		# self.block = ViTLayer(E,num_heads,mlp_ratio) #single layer for now

	def forward(self,x):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		tokens = self.block(tokens)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


class PatchEmbedding(nn.Module):
	'''
	Standard ViT patch embedding.
	Returns token tensor, i.e.: shape [B,N,E].
	'''
	def __init__(self, img_size=256,patch_size=4,in_channels=3,E=128):
		super().__init__()

		self.num_patches = (img_size // patch_size) ** 2
		self.projector   = nn.Conv2d(in_channels,E,kernel_size=patch_size,stride=patch_size)

	def forward(self, x):     # x: [B, C, H, W]
		x = self.projector(x) # [B, E, H/P, W/P]
		x = x.flatten(2)      # [B, E, N]
		x = x.transpose(1, 2) # [B, N, E]
		return x


class TokenizedDownsample(nn.Module):
	'''
	Spatial resolution downsampler. Checkerboard pattern.
	Halves height and width, doubles channels
	'''
	def __init__(self, dim):
		super().__init__()
		self.norm = nn.LayerNorm(4 * dim)
		self.reduction = nn.Linear(4*dim,2*dim)

	def forward(self, x, H, W):
		# shapes & rearrange
		B, N, E = x.shape
		x = x.view(B, H, W, E)

		# sections
		x00 = x[:, 0::2, 0::2, :] #checkerboard pattern
		x01 = x[:, 1::2, 0::2, :]
		x10 = x[:, 0::2, 1::2, :]
		x11 = x[:, 1::2, 1::2, :]

		# 4C channels
		x = torch.cat([x00, x01, x10, x11],dim=-1)
		H //= 2
		W //= 2
		x = x.view(B, H * W, 4 * E)
		x = self.norm(x)
		x = self.reduction(x) #2C

		# (B,H/2*W/2,2C)
		return x, H, W	


class TokenizedDownsampleConv(nn.Module):
	'''
	Spatial resolution downsampling. Strided convolution.
	Halves height and width, doubles channels.
	'''
	def __init__(self,E_in,H,W):
		super().__init__()
		self.H = H
		self.W = W
		self.E_in = E_in
		self.conv = nn.Conv2d(in_channels=E_in,out_channels=2*E_in,kernel_size=3,stride=2,padding=1)
		self.gelu = nn.GELU()
		self.norm = nn.LayerNorm(2*E_in)


	def forward(self,x):
		B,N,_ = x.shape
		x = x.view(B,self.H,self.W,self.E_in).permute(0,3,1,2) #->[B,C,H,W]
		x = self.gelu(self.conv(x))     #[B,2C,H/2,W/2]
		x = x.flatten(2).transpose(1,2) #[B,N/4,2E]
		x = self.norm(x)
		return x


class TokenizedUpsampleConv(nn.Module):
	def __init__(self,E_in,H,W):
		super().__init__()
		self.H = H
		self.W = W
		self.E_in = E_in

		self.conv = nn.ConvTranspose2d(E_in,E_in//2,kernel_size=4,stride=2,padding=1)
		self.gelu = nn.GELU()
		self.norm = nn.LayerNorm(E_in//2)

	def forward(self,x):
		B,N,_ = x.shape
		x = x.transpose(1,2).view(B,self.E_in,self.H,self.W)
		x = self.gelu(self.conv(x))     #[B,C/2,2H,2W]
		x = x.flatten(2).transpose(1,2) #[B,4N,E/2]
		x = self.norm(x)
		return x


class LearnedPositionalEncoding2D(nn.Module):
	'''
	Learned positional encoding of size E.
	'''
	def __init__(self, E, num_patches, dropout=0.1):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.pe      = nn.Embedding(num_patches, E) #flat embedding

	def forward(self, x):
		B, N, _ = x.shape
		positions = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
		x = x + self.pe(positions)                                 # [B, N, E]
		return self.dropout(x)


class SinusoidalPositionalEncoding2D(nn.Module):
	'''
	Fixed positional encoding (not learned). 2-dimensional sinusoidal.
	'''
	def __init__(self, E, num_patches_h, num_patches_w, dropout=0.1):
		super().__init__()
		assert E % 2 == 0, "Embedding dimension not divisible by 2"
		E_half = E // 2  # half for rows, half for cols

		self.dropout = nn.Dropout(dropout)

		# position indices for rows and cols
		row_pos = torch.arange(num_patches_h).unsqueeze(1) # [N_h, 1]
		col_pos = torch.arange(num_patches_w).unsqueeze(1) # [N_w, 1]

		div = torch.exp(torch.arange(0, E_half, 2) * -(math.log(10000.0) / E_half))  # [E_half/2,]: 16

		# row encoding: (num_patches_h, E_half)
		row_enc = torch.zeros(num_patches_h, E_half) #[64,E/2], id est [N_h,E/2]
		row_enc[:, 0::2] = torch.sin(row_pos * div)
		row_enc[:, 1::2] = torch.cos(row_pos * div)

		# col encoding: (num_patches_w, E_half)
		col_enc = torch.zeros(num_patches_w, E_half) #[64,E/2]
		col_enc[:, 0::2] = torch.sin(col_pos * div)
		col_enc[:, 1::2] = torch.cos(col_pos * div)

		# expand and concatenate over full grid
		# row_enc: (H, W, E_half) by repeating across W
		# col_enc: (H, W, E_half) by repeating across H
		row_enc = row_enc.unsqueeze(1).expand(-1, num_patches_w, -1)  #[H, W, E_half]
		col_enc = col_enc.unsqueeze(0).expand(num_patches_h, -1, -1)  #[H, W, E_half]

		# concatenate along last dim -> (H, W, E)
		pe = torch.cat([row_enc, col_enc], dim=-1)
		pe = pe.view(1, num_patches_h * num_patches_w, E) #[1, N, E]
		self.register_buffer('pe', pe)


	def forward(self, x):
		x = x + self.pe
		return self.dropout(x)


################################################################################
# CNN Encoder
################################################################################
class CNNEncoder(nn.Module):
	"""
	Five-stage CNN encoder.
	Returns [enc_1, enc_2, enc_3, enc_4] skip tensors and output enc_5 (bottleneck).

	Channels: 32 -> 64 -> 128 -> 256 -> 512
	Spatial:  H  -> H/2 -> H/4 -> H/8 -> H/16
	"""	
	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		self.encoder_1 = ConvBlock(channels,depth=cnn_layers)

		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = ConvBlock(channels*2,depth=cnn_layers)

		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)		
		self.encoder_3 = ConvBlock(channels*4,depth=cnn_layers)

		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)		
		self.encoder_4 = ConvBlock(channels*8)

		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)		
		self.encoder_5 = ConvBlock(channels*16)


	def forward(self,x):
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_1(enc_1))
		enc_3 = self.encoder_3(self.down_2(enc_2))
		enc_4 = self.encoder_4(self.down_3(enc_3))
		enc_5 = self.encoder_5(self.down_4(enc_4))		
		return [enc_1,enc_2,enc_3,enc_4], enc_5


################################################################################
# ViT Encoder 
################################################################################
class ViTEncoder(nn.Module):
	'''
	Hybrid ViT without positional encoding. 2xCNN + 3xViT layers
	'''

	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		self.encoder_1 = ConvBlock(channels=channels,depth=cnn_layers) #32
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = ConvBlock(channels*2,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=vit_layers)		
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=vit_layers)	


	def forward(self,x):
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_1(enc_1))
		enc_3 = self.encoder_3(self.down_2(enc_2))
		enc_4 = self.encoder_4(self.down_3(enc_3))
		enc_5 = self.encoder_5(self.down_4(enc_4))
		return [enc_1,enc_2,enc_3,enc_4], enc_5

################################################################################
# ViT Encoder #2
################################################################################
class ViTEncoderStemmed(nn.Module):
	'''
	Patch embedding on inputs HxW to H/4xW/4. CNN stem to pass high-dimension 
	feature maps from encoder to decoder.
	'''
	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()
		stem_params = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		#CNN STEM
		self.stem_1 = nn.Sequential(
			nn.Conv2d(32,32,**stem_params),
			nn.GroupNorm(1,32),
			nn.GELU()
		)
		self.down_1 = nn.Conv2d(32,64,**down_params)
		self.stem_2 = nn.Sequential(
			nn.Conv2d(64,64,**stem_params),
			nn.GroupNorm(1,64),
			nn.GELU()
		)

		# EMBEDDING AND POSITIONAL ENCODING
		self.embed = PatchEmbedding(img_size=256,patch_size=4,in_channels=3,E=128)
		self.pe    = SinusoidalPositionalEncoding2D(E=128,num_patches_h=64,num_patches_w=64)

		# ViT Layers
		self.encoder_3 = ViTBlock(128,num_heads=2)
		self.down_3    = nn.Conv2d(128,256,**down_params)
		self.encoder_4 = ViTBlock(256,num_heads=4)
		self.down_4    = nn.Conv2d(256,512,**down_params)		
		self.encoder_5 = ViTBlock(512,num_heads=8)


	def forward(self,x):
		B,_,_,_ = x.shape
		enc_1 = self.stem_1(x)
		enc_2 = self.stem_2(self.down_1(enc_1))

		tokens = self.pe(self.embed(x))
		bchw   = tokens.reshape(B,64,64,128).permute(0,3,1,2)

		enc_3 = self.encoder_3(bchw)
		enc_4 = self.encoder_4(self.down_3(enc_3))
		enc_5 = self.encoder_5(self.down_4(enc_4))

		return [enc_1,enc_2,enc_3,enc_4], enc_5


################################################################################
# ViT Decoder
################################################################################
class ViTDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN. Mirrors ViTEncoder().
	'''
	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()	
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}

		self.decoder_1 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ConvBlock(channels*2,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ConvBlock(channels,depth=cnn_layers)


	def forward(self,x,skips):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ))
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ))
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ))
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ))
		return dec_5


################################################################################
# CNN Decoder
################################################################################
class CNNDecoder(nn.Module):
	'''
	5-stage CNN.
	'''
	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}

		self.decoder_1 = ConvBlock(channels*16,depth=cnn_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ConvBlock(channels*8,depth=cnn_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ConvBlock(channels*4,depth=cnn_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ConvBlock(channels*2,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ConvBlock(channels,depth=cnn_layers)


	def forward(self,x,skips):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ))
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ))
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ))
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ))
		return dec_5

################################################################################
# UNET
################################################################################
_ENCODERS = {'cnn': CNNEncoder, 'vit': ViTEncoder, 'vit2': ViTEncoderStemmed}
_DECODERS = {'cnn': CNNDecoder, 'vit': ViTDecoder}

class UNet(nn.Module):
	def __init__(self,model_id,encoder='cnn',decoder='cnn',in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__()
		assert encoder in _ENCODERS, f"encoder str must be one of {list(_ENCODERS)}"
		assert decoder in _DECODERS, f"decoder str must be one of {list(_DECODERS)}"

		# PARAMS/LOGS
		self.model_id   = model_id
		self.model_name = "unet_modular"

		# LAYERS
		self.in_layer  = nn.Conv2d(in_channels,channels,3,1,1,bias=True)
		self.encoder   = _ENCODERS[encoder](cnn_layers,vit_layers,channels,mlp_ratio)
		self.decoder   = _DECODERS[decoder](cnn_layers,vit_layers,channels,mlp_ratio)
		self.out_layer = nn.Conv2d(channels,out_labels,kernel_size=1,padding=0)


	def forward(self,x):
		x             = self.in_layer(x)
		skips,enc_out = self.encoder(x)
		dec_out       = self.decoder(enc_out,skips)
		return self.out_layer(dec_out)


################################################################################
# UNET SUBCLASSES/ENTRY POINT
################################################################################
class UNet_CNN_CNN(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='cnn', decoder='cnn',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)
		self.model_name = "unet_cnn_cnn"
		self.model_id   = model_id


class UNet_CNN_ViT(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='cnn', decoder='vit',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)
		self.model_name = "unet_cnn_vit"
		self.model_id   = model_id


class UNet_ViT_CNN(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='vit', decoder='cnn',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)
		self.model_name = "unet_vit_cnn"
		self.model_id   = model_id


class UNet_ViT_ViT(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='vit', decoder='vit',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)
		self.model_name = "unet_vit_vit"
		self.model_id   = model_id


class UNet_ViT2_CNN(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='vit2', decoder='cnn',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)
		self.model_name = "unet_vit2_cnn"
		self.model_id   = model_id


class UNet_ViT2_ViT(UNet):
	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=4):
		super().__init__(model_id,encoder='vit2', decoder='vit',in_channels=in_channels, out_labels=out_labels,
			cnn_layers=cnn_layers,vit_layers=vit_layers,channels=channels,mlp_ratio=mlp_ratio)	
		self.model_name = "unet_vit2_vit"
		self.model_id   = model_id


################################################################################
# FUNCTIONS
################################################################################
def get_model_memory_size(model):

	# DUMMY INPUT
	x = torch.randn(8,3,256,256)

	# TO DEV
	model = model.cuda()
	x     = x.cuda()

	# FORWARD & BACKWARD
	out  = model(x)
	loss = out.sum()
	loss.backward()

	#PRINT
	print(f"{model.model_name}:",end=' ')
	print(torch.cuda.max_memory_allocated()/1e9,"GB")


def get_model_parameter_size(model):
	# COUNT STUFF
	all_params       = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	named_params     = {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n,m in model.named_children()}

	# (SOMEWHAT) PRETTY PRINT
	print('\n' + "-"*40)
	print(f"{model.model_name}")
	print("-"*40)
	print(f"Total:     {all_params}")
	print(f"Trainable: {trainable_params}")
	print(f"-"*40)
	for name,count in named_params.items():
		print(f"{name}: {count}")
	print("-"*40)


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	#DO SOME CHECKS
	variations = [UNet_CNN_CNN,UNet_ViT_CNN,UNet_CNN_ViT,UNet_ViT_ViT]

	for v in variations:
		kwargs = {'cnn_layers':3,'vit_layers':2,'channels':32,'mlp_ratio':4}
		model = v(model_id=999,**kwargs)

		get_model_memory_size(model)
		get_model_parameter_size(model)


		# model.eval()
		# with torch.no_grad():
			# out = model(x)
		# params = sum(p.numel() for p in model.parameters()) / 1e6
		# print(f'output: {tuple(out.shape)}  params: {params:.1f}M')
