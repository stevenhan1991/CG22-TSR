import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

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

class Encoder(nn.Module):
	def __init__(self,inc,init_channels,rb=True):
		super(Encoder,self).__init__()
		self.conv1 = nn.Conv3d(inc,init_channels,4,2,1) 
		self.conv2 = nn.Conv3d(init_channels,2*init_channels,4,2,1) 
		self.conv3 = nn.Conv3d(2*init_channels,4*init_channels,4,2,1)
		self.conv4 = nn.Conv3d(4*init_channels,8*init_channels,4,2,1) 
		self.rb = rb
		if self.rb:
			self.rb1 = ResidualBlock(8*init_channels,False,3,2,False)
			self.rb2 = ResidualBlock(8*init_channels,False,3,2,False)
			self.rb3 = ResidualBlock(8*init_channels,False,3,2,False)

	def forward(self,x):
		x1 = F.relu(self.conv1(x)) 
		x2 = F.relu(self.conv2(x1))
		x3 = F.relu(self.conv3(x2))
		x4 = F.relu(self.conv4(x3))
		if self.rb:
			x4 = self.rb1(x4)
			x4 = self.rb2(x4)
			x4 = self.rb3(x4)
		return [x1,x2,x3,x4]

class Decoder(nn.Module):
	def __init__(self,outc,init_channels,ac,num):
		super(Decoder,self).__init__()
		self.deconv41 = nn.ConvTranspose3d(init_channels,init_channels//2,4,2,1) 
		self.conv_u41 = nn.Conv3d(init_channels,init_channels//2,3,1,1)
		self.deconv31 = nn.ConvTranspose3d(init_channels//2,init_channels//4,4,2,1) 
		self.conv_u31 = nn.Conv3d(init_channels//2,init_channels//4,3,1,1)
		self.deconv21 = nn.ConvTranspose3d(init_channels//4,init_channels//8,4,2,1)
		self.conv_u21 = nn.Conv3d(init_channels//4,init_channels//8,3,1,1)
		self.deconv11 = nn.ConvTranspose3d(init_channels//8,init_channels//16,4,2,1)
		self.conv_u11 = nn.Conv3d(init_channels//16,outc,3,1,1)
		'''
		self.decoders = nn.Sequential()
		self.num = num
		for k in range(0,num):
			decoder = nn.Sequential(OrderedDict({'deconv21': nn.ConvTranspose3d(init_channels//4,init_channels//8,4,2,1),
				                       'conv_u21': nn.Conv3d(init_channels//4,init_channels//8,3,1,1),
				                       'deconv11': nn.ConvTranspose3d(init_channels//8,init_channels//16,4,2,1),
				                       'convu_11': nn.Conv3d(init_channels//16,outc,3,1,1)}
				                       ))
			self.decoders.add_module("decoder"+str(k), decoder)
		'''
		self.ac = ac

	def forward(self,features):
		u11 = F.relu(self.deconv41(features[-1]))
		u11 = F.relu(self.conv_u41(torch.cat((features[-2],u11),dim=1)))
		u21 = F.relu(self.deconv31(u11))
		u21 = F.relu(self.conv_u31(torch.cat((features[-3],u21),dim=1)))
		u31 = F.relu(self.deconv21(u21)) 
		u31 = F.relu(self.conv_u21(torch.cat((features[-4],u31),dim=1)))
		u41 = F.relu(self.deconv11(u31))
		out = self.conv_u11(u41)
		'''
		results = []
		for k in range(0,self.num):
			decoder = self.decoders._modules["decoder"+str(k)]
			u31 = F.relu(decoder._modules["deconv21"](u21))
			u31 = F.relu(decoder._modules["conv_u21"](torch.cat((features[-4],u31),dim=1)))
			u41 = F.relu(decoder._modules["deconv11"](u31))
			out = decoder._modules["convu_11"](u41)
			results.append(out)
			if self.ac :
				out = self.ac(out)
		out = torch.stack(results)
		out = torch.squeeze(out)
		return torch.unsqueeze(out,0)
		'''
		return out


class UNet(nn.Module):
	def __init__(self,inc,outc,init_channels,num,ac=None,rb=True):
		super(UNet,self).__init__()
		self.encoder = Encoder(inc,init_channels,rb)
		self.decoder = Decoder(outc,init_channels*8,ac,num)

	def forward(self,x):
		return self.decoder(self.encoder(x))