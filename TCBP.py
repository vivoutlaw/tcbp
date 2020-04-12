import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class TCBP(torch.nn.Module):
	"""
	TCBP layer implementation
	# https://arxiv.org/pdf/2004.02205.pdf

	Example:
	from TCBP import TCBP
	import torch
	x = torch.rand([10,512,4,7,7]) 
	tcbp = TCBP(input_dim1=512, input_dim2=512,output_dim=256, temporal_window=4, spat_x=7, spat_y=7)
	t =tcbp(x,x)
	t.shape
	---> torch.Size([10, 256])

	"""
	def __init__(self, input_dim1, input_dim2, output_dim, temporal_window, spat_x, spat_y, sum_pool = True):
		super(TCBP, self).__init__()
		self.output_dim = output_dim
		self.sum_pool = sum_pool
		self.T = temporal_window
		self.spat_x = spat_x
		self.spat_y = spat_y
		rand_h_1 =  torch.randint(output_dim, size = (input_dim1,)).repeat(self.T,1).transpose(0,1).contiguous().view(-1)
		rand_h_2 = torch.randint(output_dim, size = (input_dim2,)).repeat(self.T,1).transpose(0,1).contiguous().view(-1)
		generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
		self.sketch_matrix1 = torch.nn.Parameter(generate_sketch_matrix(rand_h_1, 2 * torch.randint(2, size = (input_dim1*self.T,)) - 1, input_dim1*self.T, output_dim))
		self.sketch_matrix2 = torch.nn.Parameter(generate_sketch_matrix(rand_h_2, 2 * torch.randint(2, size = (input_dim2*self.T,)) - 1, input_dim2*self.T, output_dim))

	def forward(self, x1, x2):
		x1  = F.avg_pool3d(x1, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x1.shape[0],-1)
		x2  = F.avg_pool3d(x2, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x2.shape[0],-1)
		fft1 = torch.rfft(x1.matmul(self.sketch_matrix1), 1)
		fft2 = torch.rfft(x2.matmul(self.sketch_matrix2), 1)
		fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
		out = torch.irfft(fft_product, 1, signal_sizes = (self.output_dim,)) * self.output_dim	

		# signed sqrt, L2 normalization
		out = torch.mul(torch.sign(out),torch.sqrt(torch.abs(out)+1e-12))  # signed sqrt
		out = F.normalize(out, p=2, dim=1)	# L2 normalize
		return out

