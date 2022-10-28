import numpy as np
import torch


class DataSet():
	def __init__(self,args):
		self.path = args.data_path
		self.dataset = args.dataset
		if self.dataset == 'plume':
			self.dim = [64,64,256]
			self.c = [64,64,128]
			self.total_samples = 27
		elif self.dataset == 'tornado':
			self.dim = [128,128,128]
			self.c = [96,96,96]
			self.total_samples = 48
		elif self.dataset == 'hurricane':
			self.dim = [256,256,56]
			self.c = [128,128,48]
			self.total_samples = 48
		elif self.dataset == 'supernova':
			self.dim = [128,128,128]
			self.c = [64,64,64]
			self.total_samples = 100
		elif self.dataset == 'cylinder':
			self.dim = [640,240,80]
			self.c = [80,80,80]
			self.total_samples = 100
		self.training_samples = args.samples
		self.interval = args.interval
		self.data = np.zeros((self.total_samples,3,self.dim[0],self.dim[1],self.dim[2]))
		self.croptimes = args.croptimes
		if (self.dim[0] == self.c[0]) and (self.dim[1] == self.c[1]) and (self.dim[2] == self.c[2]):
			self.croptimes = 1

	def ReadData(self):
		for i in range(1,self.total_samples+1):
			print(i)
			v = np.fromfile('/afs/crc.nd.edu/user/j/jhan5/vis/SciVis19/vector_data_wo_norm/half-cylinder/320/half-cylinder'+'{:03d}'.format(i)+'.vec',dtype='<f')
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0],3).transpose()
			self.data[i-1] = v

	def GetTrainingData(self):
		group = self.training_samples-self.interval-2
		s = np.zeros((self.croptimes*group,1,3,self.c[0],self.c[1],self.c[2]))
		i = np.zeros((self.croptimes*group,self.interval,3,self.c[0],self.c[1],self.c[2]))
		e = np.zeros((self.croptimes*group,1,3,self.c[0],self.c[1],self.c[2]))
		idx = 0
		for k in range(0,group):
			sc,ic,ec = self.CropData(self.data[k:k+self.interval+2])
			for j in range(0,self.croptimes):
				s[idx] = sc[j]
				i[idx] = ic[j]
				e[idx] = ec[j]
				idx += 1
		s = torch.FloatTensor(s)
		i = torch.FloatTensor(i)
		e = torch.FloatTensor(e)
		return s,i,e

	def CropData(self,data):
		s = []
		i = []
		e = []
		n = 0
		while n<self.croptimes:
			if self.c[0]==self.dim[0]:
				x = 0
			else:
				x = np.random.randint(0,self.dim[0]-self.c[0])
			if self.c[1] == self.dim[1]:
				y = 0
			else:
				y = np.random.randint(0,self.dim[1]-self.c[1])
			if self.c[2] == self.dim[2]:
				z = 0
			else:
				z = np.random.randint(0,self.dim[2]-self.c[2])
			sc = data[0:1,0:3,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			ic = data[1:1+self.interval,0:3,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			ec = data[1+self.interval:,0:3,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			s.append(sc)
			i.append(ic)
			e.append(ec)
			n = n+1
		return s,i,e


