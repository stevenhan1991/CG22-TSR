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

class FCN(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(FCN,self).__init__()
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


class InterpolationNet(nn.Module):
	def __init__(self,inc,outc,init_channels,num,rb=True):
		super(InterpolationNet,self).__init__()
		self.U = UNet(inc,outc,init_channels,num,rb)
		self.V = UNet(inc,outc,init_channels,num,rb)
		self.W = UNet(inc,outc,init_channels,num,rb)
		self.interval = outc

	def forward(self,s,e):
		u = torch.cat((s[:,0:1,:,:,:,],e[:,0:1,:,:,:,]),dim=1)
		v = torch.cat((s[:,1:2,:,:,:,],e[:,1:2,:,:,:,]),dim=1)
		w = torch.cat((s[:,2:3,:,:,:,],e[:,2:3,:,:,:,]),dim=1)
		U = self.U(u)
		V = self.V(v)
		W = self.W(w)
		return U,V,W
		'''
		vecs = []
		for i in range(0,self.interval):
			vec = torch.cat((U[:,i:i+1,:,:,:,],V[:,i:i+1,:,:,:,],W[:,i:i+1,:,:,:,]),dim=1)
			vecs.append(vec)
		vecs = torch.stack(vecs)
		return torch.squeeze(vecs)
		'''

class RInterpolationNet(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(RInterpolationNet,self).__init__()
		self.U = FCN(inc,outc,init_channels,num)
		self.V = FCN(inc,outc,init_channels,num)
		self.W = FCN(inc,outc,init_channels,num)
		self.interval = num

	def forward(self,s,e):
		u = torch.cat((s[:,0:1,:,:,:,],e[:,0:1,:,:,:,]),dim=1)
		v = torch.cat((s[:,1:2,:,:,:,],e[:,1:2,:,:,:,]),dim=1)
		w = torch.cat((s[:,2:3,:,:,:,],e[:,2:3,:,:,:,]),dim=1)
		U = self.U(u)
		V = self.V(v)
		W = self.W(w)
		return U,V,W

class FCNwoLSTM(nn.Module):
	def __init__(self,inc,outc,init_channels):
		super(FCNwoLSTM,self).__init__()
		self.encoder = nn.Sequential(*[nn.Conv3d(inc,init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(init_channels,2*init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(2*init_channels,4*init_channels,4,2,1),
			                            nn.ReLU(inplace=True),
			                            nn.Conv3d(4*init_channels,8*init_channels,4,2,1),
			                            nn.ReLU(inplace=True)
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

	def forward(self,x):
		return torch.sigmoid(self.decoder(self.encoder(x)))

class MaskNet(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(MaskNet,self).__init__()
		self.U = UNet(inc,outc,init_channels,torch.sigmoid,num,False)
		self.V = UNet(inc,outc,init_channels,torch.sigmoid,num,False)
		self.W = UNet(inc,outc,init_channels,torch.sigmoid,num,False)
		self.interval = num

	def forward(self,s,e):
		u = torch.cat((s[:,0:1,:,:,:,],e[:,0:1,:,:,:,]),dim=1)
		v = torch.cat((s[:,1:2,:,:,:,],e[:,1:2,:,:,:,]),dim=1)
		w = torch.cat((s[:,2:3,:,:,:,],e[:,2:3,:,:,:,]),dim=1)
		M_u = self.U(u)
		M_v = self.V(v)
		M_w = self.W(w)
		return M_u,M_v,M_w

class RMaskNet(nn.Module):
	def __init__(self,inc,outc,init_channels):
		super(RMaskNet,self).__init__()
		self.U = FCNwoLSTM(inc,outc,init_channels)
		self.V = FCNwoLSTM(inc,outc,init_channels)
		self.W = FCNwoLSTM(inc,outc,init_channels)
		self.interval = outc

	def forward(self,s,e):
		u = torch.cat((s[:,0:1,:,:,:,],e[:,0:1,:,:,:,]),dim=1)
		v = torch.cat((s[:,1:2,:,:,:,],e[:,1:2,:,:,:,]),dim=1)
		w = torch.cat((s[:,2:3,:,:,:,],e[:,2:3,:,:,:,]),dim=1)
		M_u = self.U(u)
		M_v = self.V(v)
		M_w = self.W(w)
		return M_u,M_v,M_w

class TSRVFD(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(TSRVFD,self).__init__()
		self.Interpolation = InterpolationNet(inc,outc,init_channels,num)
		self.Mask = MaskNet(inc,outc,init_channels,num)
		self.interval = outc

	def forward(self,s,e):
		U,V,W = self.Interpolation(s,e)
		M_u,M_v,M_w = self.Mask(s,e)
		vecs = []
		for i in range(1,self.interval+1):
			weight = i/(self.interval+1)
			linear = weight*e+(1-weight)*s
			vec_U = M_u[:,i-1:i,:,:,:,]*U[:,i-1:i,:,:,:,]+(1-M_u[:,i-1:i,:,:,:,])*linear[:,0:1,:,:,:,]
			vec_V = M_v[:,i-1:i,:,:,:,]*V[:,i-1:i,:,:,:,]+(1-M_v[:,i-1:i,:,:,:,])*linear[:,1:2,:,:,:,]
			vec_W = M_w[:,i-1:i,:,:,:,]*W[:,i-1:i,:,:,:,]+(1-M_w[:,i-1:i,:,:,:,])*linear[:,2:3,:,:,:,]
			vec = torch.cat((vec_U,vec_V,vec_W),dim=1)
			vecs.append(vec)
		vecs = torch.stack(vecs)
		return torch.squeeze(vecs)

class TSRVFDwoSep(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(TSRVFDwoSep,self).__init__()
		self.Interpolation = UNet(3*inc,3*outc,init_channels,num,ac=None,rb=True)
		self.Mask = UNet(3*inc,3*outc,init_channels,num,ac=F.sigmoid,rb=False)
		self.interval = outc

	def forward(self,s,e):
		p = torch.cat((s,e),dim=1)
		vecs = self.Interpolation(p)
		masks = self.Mask(p)
		v = []
		for i in range(1,self.interval+1):
			weight = i/(self.interval+1)
			linear = weight*e+(1-weight)*s
			v_ = masks[:,3*(i-1):3*i,:,:,:,]*vecs[:,3*(i-1):3*i,:,:,:,]+(1-masks[:,3*(i-1):3*i,:,:,:,])*linear
			v.append(v_)
		v = torch.stack(v)
		return torch.squeeze(v) 



class RTSRVFD(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(RTSRVFD,self).__init__()
		self.Interpolation = RInterpolationNet(inc,outc,init_channels,num)
		self.Mask = RMaskNet(inc,num,init_channels)
		self.interval = num

	def forward(self,s,e):
		U,V,W = self.Interpolation(s,e)
		M_u,M_v,M_w = self.Mask(s,e)
		vecs = []
		for i in range(1,self.interval+1):
			weight = i/(self.interval+1)
			linear = weight*e+(1-weight)*s
			vec_U = M_u[:,i-1:i,:,:,:,]*U[:,i-1:i,:,:,:,]+(1-M_u[:,i-1:i,:,:,:,])*linear[:,0:1,:,:,:,]
			vec_V = M_v[:,i-1:i,:,:,:,]*V[:,i-1:i,:,:,:,]+(1-M_v[:,i-1:i,:,:,:,])*linear[:,1:2,:,:,:,]
			vec_W = M_w[:,i-1:i,:,:,:,]*W[:,i-1:i,:,:,:,]+(1-M_w[:,i-1:i,:,:,:,])*linear[:,2:3,:,:,:,]
			vec = torch.cat((vec_U,vec_V,vec_W),dim=1)
			vecs.append(vec)
		vecs = torch.stack(vecs)
		return torch.squeeze(vecs)


class UncentaintyNet(nn.Module):
	def __init__(self):
		super(UncentaintyNet,self).__init__()
		self.net = nn.Sequential(*[nn.Conv3d(1,16,3,1,1),
			                       nn.ReLU(inplace=True),
			                       nn.Conv3d(16,1,3,1,1)
			                       ])
	def forward(self,x):
		return F.sigmoid(self.net(x))


'''
class TSRVFD(nn.Module):
	def __init__(self,inc,outc,init_channels,num):
		super(TSRVFD,self).__init__()
		self.Interpolation = InterpolationNet(inc,outc,init_channels,num)
		self.UncentaintyU = UncentaintyNet()
		self.UncentaintyV = UncentaintyNet()
		self.UncentaintyW = UncentaintyNet()
		self.UncentaintyLU = UncentaintyNet()
		self.UncentaintyLV = UncentaintyNet()
		self.UncentaintyLW = UncentaintyNet()
		self.interval = outc
		self.softmax = nn.Softmax(dim=1)

	def forward(self,s,e):
		results = []
		U,V,W = self.Interpolation(s,e)
		UU = self.UncentaintyU(U.permute(1,0,2,3,4)).permute(1,0,2,3,4) ###[1,5,L,H,W]
		UV = self.UncentaintyV(V.permute(1,0,2,3,4)).permute(1,0,2,3,4)
		UW = self.UncentaintyW(W.permute(1,0,2,3,4)).permute(1,0,2,3,4)
		lerp = []
		for j in range(1,self.interval+1):
			weight = j/(self.interval+1)
			vec = weight*e+(1-weight)*s
			lerp.append(vec)
		lerp = torch.squeeze(torch.stack(lerp)) # [5,3,L,H,W]
		ULU = self.UncentaintyLU(lerp[:,0:1,:,:,:,]).permute(1,0,2,3,4) ## [1,5,L,H,W]
		ULV = self.UncentaintyLV(lerp[:,1:2,:,:,:,]).permute(1,0,2,3,4)
		ULW = self.UncentaintyLW(lerp[:,2:3,:,:,:,]).permute(1,0,2,3,4)
		UU_new = 1/UU
		UV_new = 1/UV
		UW_new = 1/UW
		ULU_new = 1/ULU
		ULV_new = 1/ULV
		ULW_new = 1/ULW
		vecs = []
		uncens = []
		uncens_lerp = []
		uncens_lerp_new = []
		uncens_dl_new = []
		for i in range(0,self.interval):
			vec = torch.cat((U[:,i:i+1,:,:,:,],V[:,i:i+1,:,:,:,],W[:,i:i+1,:,:,:,]),dim=1)
			uncen = torch.cat((UU[:,i:i+1,:,:,:,],UV[:,i:i+1,:,:,:,],UW[:,i:i+1,:,:,:,]),dim=1)
			uncen_lerp = torch.cat((ULU[:,i:i+1,:,:,:,],ULV[:,i:i+1,:,:,:,],ULW[:,i:i+1,:,:,:,]),dim=1)
			uncen_combine_u = self.softmax(torch.cat((UU_new[:,i:i+1,:,:,:,],ULU_new[:,i:i+1,:,:,:,]),dim=1))
			uncen_combine_v = self.softmax(torch.cat((UV_new[:,i:i+1,:,:,:,],ULV_new[:,i:i+1,:,:,:,]),dim=1))
			uncen_combine_w = self.softmax(torch.cat((UW_new[:,i:i+1,:,:,:,],ULW_new[:,i:i+1,:,:,:,]),dim=1))
			uncen_dl_new = torch.cat((uncen_combine_u[:,0:1,:,:,:,],uncen_combine_v[:,0:1,:,:,:,],uncen_combine_w[:,0:1,:,:,:,]),dim=1)
			uncen_lerp_new = torch.cat((uncen_combine_u[:,1:2,:,:,:,],uncen_combine_v[:,1:2,:,:,:,],uncen_combine_w[:,1:2,:,:,:,]),dim=1)
			vecs.append(vec)
			uncens.append(uncen)
			uncens_lerp.append(uncen_lerp)
			uncens_dl_new.append(uncen_dl_new)
			uncens_lerp_new.append(uncen_lerp_new)
		vecs = torch.squeeze(torch.stack(vecs))
		uncens = torch.squeeze(torch.stack(uncens))
		uncens_lerp = torch.squeeze(torch.stack(uncens_lerp))
		uncens_dl_new = torch.squeeze(torch.stack(uncens_dl_new))
		uncens_lerp_new = torch.squeeze(torch.stack(uncens_lerp_new))
		combined = vecs*uncens_dl_new+lerp*uncens_lerp_new
		return vecs,lerp,uncens,uncens_lerp,combined
'''

class Dis(nn.Module):
    def __init__(self):
        super(Dis,self).__init__()
        ### downsample 
        self.conv1 = nn.Conv3d(3,32,4,2,1)
        self.conv2 = nn.Conv3d(32,64,4,2,1)
        self.conv3 = nn.Conv3d(64,128,4,2,1)
        self.conv4 = nn.Conv3d(128,1,4,2,1)
        self.ac = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        x1 = self.ac(self.conv1(x)) 
        x2 = self.ac(self.conv2(x1))
        x3 = self.ac(self.conv3(x2))
        x4 = self.ac(self.conv4(x3))
        x5 = F.avg_pool3d(x4,x4.size()[2:])
        return x5.view(-1)

class RelMSE(nn.Module):
	def __init__(self,eps=1e-6):
		super(RelMSE,self).__init__()
		self.eps = eps

	def forward(self,s,g):
		diff = torch.mean(torch.abs(s-g)/(torch.abs(g)))
		return diff

class UncentaintyLoss(nn.Module):
	def __init__(self):
		super(UncentaintyLoss,self).__init__()

	def forward(self,gt,syn,uncen):
		loss = torch.abs(syn-gt)/((2*uncen*uncen)*(torch.abs(gt)+1e-6))+torch.log(2*uncen*uncen)+1/(2.0*math.pi)
		return torch.mean(loss)






