import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math

def BuildResidualBlock(channels,dropout,kernel,depth,bias):
  layers = []
  for i in range(int(depth)):
    layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
               #nn.BatchNorm3d(channels),
               nn.ReLU(True)]
    if dropout:
      layers += [nn.Dropout(0.5)]
  layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
             #nn.BatchNorm3d(channels),
           ]
  return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
  def __init__(self,channels,dropout,kernel,depth,bias):
    super(ResidualBlock,self).__init__()
    self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

  def forward(self,x):
    out = x+self.block(x)
    return out

class LSTMCell(nn.Module):
	def __init__(self,input_size,hidden_size,kernel):
		super(LSTMCell,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		pad = kernel//2
		self.Gates = nn.Conv3d(input_size+hidden_size,4*hidden_size,kernel,padding=pad)

	def forward(self,input_,prev_hidden=None,prev_cell=None):
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]
		if prev_hidden is None and prev_cell is None:
			state_size = [batch_size,self.hidden_size]+list(spatial_size)
			prev_hidden = torch.zeros(state_size)
			prev_cell = torch.zeros(state_size)
		prev_hidden = prev_hidden.cuda()
		prev_cell = prev_cell.cuda()
		stacked_inputs = torch.cat((input_,prev_hidden),1)
		gates = self.Gates(stacked_inputs)

		in_gate,remember_gate,out_gate,cell_gate = gates.chunk(4,1)
		in_gate = torch.sigmoid(in_gate)
		remember_gate = torch.sigmoid(remember_gate)
		out_gate = torch.sigmoid(out_gate)
		cell_gate = torch.tanh(cell_gate)

		cell = (remember_gate*prev_cell)+(in_gate*cell_gate)
		hidden = out_gate*torch.tanh(cell)
		return hidden,cell


class TSR(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(TSR,self).__init__()
		self.encoder = nn.Sequential(*[nn.Conv3d(inc,init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(init_channels,2*init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(2*init_channels,4*init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(4*init_channels,8*init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            ResidualBlock(8*init_channels,False,3,2,False),
			                            ResidualBlock(8*init_channels,False,3,2,False),
			                            ResidualBlock(8*init_channels,False,3,2,False)
			                            ])

		self.decoder = nn.Sequential(*[nn.ConvTranspose3d(init_channels*8,init_channels*4,4,2,1),
			                           nn.ReLU(inplace=True),
			                           nn.ConvTranspose3d(init_channels*4,init_channels*2,4,2,1),
			                           nn.ReLU(inplace=True),
			                           nn.ConvTranspose3d(init_channels*2,init_channels,4,2,1),
			                           nn.ReLU(inplace=True),
			                           nn.ConvTranspose3d(init_channels,init_channels//2,4,2,1),
			                           nn.ReLU(inplace=True),
			                           nn.Conv3d(init_channels//2,outc,3,1,1)
			                           ])
		self.interval = num
		self.lstm = LSTMCell(init_channels*8,init_channels*8,3)

	def forward(self,x):
		x = self.encoder(x)
		h = None
		c = None
		comps = []
		for i in range(self.interval):
			h,c = self.lstm(x,h,c)
			comp = self.decoder(h)
			comps.append(comp)
		comps = torch.stack(comps)
		comps = comps.permute(1,2,0,3,4,5)
		comps = torch.squeeze(comps,0)
		return comps