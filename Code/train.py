import torch.nn as nn
import torch.optim as optim
import time
import argparse
import DataPre
import torch
import numpy as np
from model import RelMSE, UncentaintyLoss
from torch.autograd import Variable


def trainInterpolationNet(model,args,dataset):
	optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999))
	criterion = nn.MSELoss()
	for itera in range(1,args.epochs+1):
		print('======='+str(itera)+'========')
		s,i,e = dataset.GetTrainingData()
		loss_mse = 0
		x = time.time()
		for k in range(0,s.size()[0]):
			starting = s[k]
			ending = e[k]
			intermerdiate = i[k]
			if args.cuda:
				starting = starting.cuda()
				ending = ending.cuda()
				intermerdiate = intermerdiate.cuda()
			optimizer.zero_grad()
			syn =  model(starting,ending)
			error = criterion(syn,intermerdiate)
			error.backward()
			loss_mse += error.mean().item()
			optimizer.step()
		y = time.time()
		print("Time = "+str(y-x))
		print("Loss = "+str(loss_mse))
		if itera%100==0 or itera==1:
			torch.save(model,args.model_path+args.dataset+'/Interpolation-'+str(itera)+'.pth')

def trainMaskNet(model,Interpolation,args,dataset):
	optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.5,0.999))
	if args.loss == 'mse':
		criterion = nn.MSELoss()
	elif args.loss == 'relmse':
		criterion = RelMSE()
	for itera in range(1,args.epochs+1):
		print('======='+str(itera)+'========')
		for m in range(len(dataset)):
			s,i,e = dataset[m].GetTrainingData()
			loss_mse = 0
			x = time.time()
			for k in range(0,s.size()[0]):
				starting = s[k]
				ending = e[k]
				intermerdiate = i[k]
				if args.cuda:
					starting = starting.cuda()
					ending = ending.cuda()
					intermerdiate = intermerdiate.cuda()
				optimizer.zero_grad()
				syn =  Interpolation(starting,ending)
				masks = model(starting,ending)
				error = 0
				for j in range(1,args.interval+1):
					weight = j/(args.interval+1)
					linear = weight*ending+(1-weight)*starting
					vec = masks[j-1:j,:,:,:,:,]*syn[j-1:j,:,:,:,:,]+(1-masks[j-1:j,:,:,:,:,]*linear)
					#vec = linear
					error += criterion(vec,intermerdiate[j-1:j,:,:,:,:,])
				error.backward()
				loss_mse += error.mean().item()
				optimizer.step()
		y = time.time()
		print("Time = "+str(y-x))
		print("Loss = "+str(loss_mse))
		if itera%100==0 or itera==1:
			torch.save(model,args.model_path+args.dataset+'/Mask-'+str(itera)+'.pth')


def trainTSRVFD(model,args,dataset):
	optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999))
	criterion = nn.MSELoss()
	criterion_uncen = UncentaintyLoss()
	for itera in range(1,args.epochs+1):
		print('======='+str(itera)+'========')
		s,i,e = dataset.GetTrainingData()
		loss_mse = 0
		loss = 0
		x = time.time()
		for k in range(0,s.size()[0]):
			starting = s[k]
			ending = e[k]
			intermerdiate = i[k]
			if args.cuda:
				starting = starting.cuda()
				ending = ending.cuda()
				intermerdiate = intermerdiate.cuda()
			optimizer.zero_grad()
			syns =  model(starting,ending)
			error = criterion(syns,intermerdiate)#+criterion(combined,intermerdiate)+criterion_uncen(intermerdiate,syns,uncens)+criterion_uncen(intermerdiate,lerp,uncens_lerp)
			error.backward()
			loss += error.item()
			#loss_mse += criterion(combined,intermerdiate).item()
			optimizer.step()
		y = time.time()
		print("Time = "+str(y-x))
		print("Total Loss = "+str(loss))
		#print("MSE Loss = "+str(loss_mse))
		if itera%100==0 or itera==1:
			torch.save(model,args.model_path+args.dataset+'/TSR-wo-sep-'+str(itera)+'.pth')


def trainGAN(G,D,args,dataset):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4,betas=(0.5,0.999)) #betas=(0.5,0.999)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4,betas=(0.5,0.999))
    ganloss = nn.MSELoss()#nn.BCEWithLogitsLoss()#GANLoss(args.ganloss)
    critic = 1
    for itera in range(1,args.epochs+1):
        s,i,e = dataset.GetTrainingData()
        loss_G = 0
        loss_D = 0
        mse_loss = 0
        print("========================")
        print(itera)
        x = time.time()
        for k in range(0,s.size()[0]):
        	starting = s[k]
        	ending = e[k]
        	intermerdiate = i[k]
        	if args.cuda:
        		starting = starting.cuda()
        		ending = ending.cuda()
        		intermerdiate = intermerdiate.cuda()
        	batch = intermerdiate.size()[0]
        	for p in G.parameters():
        		p.requires_grad = False
        	for j in range(1,critic+1):
        		optimizer_D.zero_grad()
        		label_real = Variable(torch.full((batch,),1.0,device=device))
        		output_real = D(intermerdiate)
        		real_loss = ganloss(output_real,label_real)
        		fake_data = G(starting,ending)
        		label_fake = Variable(torch.full((batch,),0.0,device=device))
        		output_fake = D(fake_data)
        		fake_loss = ganloss(output_fake,label_fake)
        		loss = 0.5*(real_loss+fake_loss)
        		loss.backward()
        		loss_D += loss.mean().item()
        		optimizer_D.step()
        	for p in G.parameters():
        		p.requires_grad = True
        	for p in D.parameters():
        		p.requires_grad = False
        	for j in range(1,1+1):
        		optimizer_G.zero_grad()
        		label_real = Variable(torch.full((batch,),1.0,device=device))
        		fake_data = G(starting,ending)
        		output_real = D(fake_data)
        		L_adv = ganloss(output_real,label_real)
        		L_c = ganloss(fake_data,intermerdiate)
        		error = 1e-2*L_adv+1*L_c
        		mse_loss += L_c.item()
        		error.backward()
        		loss_G += error.item()
        		optimizer_G.step()
        	for p in D.parameters():
        		p.requires_grad = True
        y = time.time()
        print('Loss G = '+str(loss_G))
        print('Loss D = '+str(loss_D))
        print('Loss MSE = '+str(mse_loss))
        print('Time ='+str(y-x))
        if itera%100 == 0 or itera==1:
        	torch.save(G,args.model_path+args.dataset+'/GAN-'+str(itera)+'.pth')

def inference(model,Vec,args):
	for i in range(0,len(Vec.data),args.interval+1):
		s = np.zeros((1,3,Vec.dim[0],Vec.dim[1],Vec.dim[2]))
		e = np.zeros((1,3,Vec.dim[0],Vec.dim[1],Vec.dim[2]))
		if (i+args.interval+1)<len(Vec.data):
			s[0] = Vec.data[i]
			e[0] = Vec.data[i+args.interval+1]
			s = torch.FloatTensor(s).cuda()
			e = torch.FloatTensor(e).cuda()
			x = time.time()
			if args.dataset != 'hurricane':
				with torch.no_grad():
					intermerdiate = model(s,e)
					intermerdiate = intermerdiate.cpu().detach().numpy()
			else:
				intermerdiate = concatsubvolume(model,[s,e],[256,256,48],args)
				s = s.cpu().detach().numpy()
				s[s!=0] = 1
				intermerdiate = intermerdiate*s
			y = time.time()
			print((y-x)/args.interval)
			for j in range(1,args.interval+1):
				data = intermerdiate[j-1]
				data = np.asarray(data,dtype='<f')
				data = data.flatten('F')
				data.tofile(args.result_path+args.mode+'/'+args.dataset+'{:03d}'.format(i+j+1)+'.vec',format='<f')

def concatsubvolume(model,data,win_size,args):
	x,y,z = data[0].size()[2],data[0].size()[3],data[0].size()[4]
	w = np.zeros((win_size[0],win_size[1],win_size[2]))
	for i in range(win_size[0]):
		for j in range(win_size[1]):
			for k in range(win_size[2]):
				dx = min(i,win_size[0]-1-i)
				dy = min(j,win_size[1]-1-j)
				dz = min(k,win_size[2]-1-k)
				d = min(min(dx,dy),dz)+1
				w[i,j,k] = d
	w = w/np.max(w)
	avI = np.zeros((x,y,z))
	pmap= np.zeros((args.interval,3,x,y,z))
	avk = 4
	for i in range((avk*x-win_size[0])//win_size[0]+1):
		for j in range((avk*y-win_size[1])//win_size[1]+1):
			for k in range((avk*z-win_size[2])//win_size[2]+1):
				si = (i*win_size[0]//avk)
				ei = si+win_size[0]
				sj = (j*win_size[1]//avk)
				ej = sj+win_size[1]
				sk = (k*win_size[2]//avk)
				ek = sk+win_size[2]
				if ei>x:
					ei= x
					si=ei-win_size[0]
				if ej>y:
					ej = y
					sj = ej-win_size[1]
				if ek>z:
					ek = z
					sk = ek-win_size[2]
				d0 = data[0][:,:,si:ei,sj:ej,sk:ek]
				d1 = data[1][:,:,si:ei,sj:ej,sk:ek]
				with torch.no_grad():
					intermerdiate = model(d0,d1)
				k = np.multiply(intermerdiate.cpu().detach().numpy(),w)
				avI[si:ei,sj:ej,sk:ek] += w
				pmap[:,:,si:ei,sj:ej,sk:ek] += k
	result = np.divide(pmap,avI)
	return result

