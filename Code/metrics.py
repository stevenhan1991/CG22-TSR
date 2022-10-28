import numpy as np
'''
from skimage.transform import resize
from skimage.measure import compare_psnr,compare_nrmse,compare_mse
from skimage.measure import compare_ssim
from skimage.io import imread
from skimage import data,img_as_float
'''
import time
import torch
from math import pi

def getSSIM(file1,file2):
	t = img_as_float(imread(file1))
	m = np.max(t)-np.min(t)
	v = img_as_float(imread(file2))
	ssim = compare_ssim(t,v[:,:,0:3],win_size=27,multichannel=True)
	return ssim

def getPSNR(t,v):
	m = np.max(t)-np.min(t)
	mse = np.mean((t-v)**2)
	psnr = 20*np.log10(m)-10*np.log10(mse)
	return psnr



def PSNR(file1,file2,outfile,start,end,Name):
	psnr = open(outfile,'w')
	avg_psnr = 0
	for i in range(start,end+1):
		print(i)
		gt = np.fromfile(file1+inttostring(i)+'.dat',dtype='<f')
		gt = 2*gt/255-1
		if Name == 'Bicubic':
			v = np.fromfile(file2+inttostring(i)+'.dat',dtype='<f')
			v = 2*v/255-1
		else:
			v = np.fromfile(file2+inttostring(i)+Name+'.dat',dtype='<f')
		p = getPSNR(gt,v)
		avg_psnr += p
		psnr.write(str(p))
		psnr.write('\t')
	psnr.close()
	print(avg_psnr/(end-start+1))

def SSIM(file1,file2,outfile,start,end):
	ssim = open(outfile,'w')
	avg_ssim = 0
	for i in range(start,end+1):
		p = getSSIM(file1+inttostring(i)+'.png',file2+inttostring(i)+'.png')
		avg_ssim += p
		ssim.write(str(p))
		ssim.write('\t')
	ssim.close()
	print(avg_ssim/(end-start+1))

def MSEVolume(file1,file2,outfile):
	gt = np.fromfile(file1,dtype='<f')
	synthesized = np.fromfile(file2,dtype='<f')
	diff = (gt-synthesized)**2
	MSE = [np.sqrt(np.sum(diff[i:i+3])) for i in range(0,len(diff),3)]
	print(np.max(MSE))
	print(np.min(MSE))
	MSE = np.asarray(MSE,dtype='<f')
	MSE.tofile(outfile,format='<f')

def AngleVolume(file1,file2,outfile):
	gt = np.fromfile(file1,dtype='<f')
	synthesized = np.fromfile(file2,dtype='<f')
	cos_angle = [cosine(gt[i:i+3],synthesized[i:i+3]) for i in range(0,len(gt),3)]
	cos_angle = np.asarray(cos_angle)
	cos_angle.tofile(outfile,format='<f')

def getAAD(t,v):
	
	t = torch.FloatTensor(t)
	v = torch.FloatTensor(v)
	cos = torch.sum(t*v,dim=0) / (torch.norm(t, dim=0) * torch.norm(v, dim=0) + 1e-10)
	cos[cos>1] = 1
	cos[cos<-1] = -1
	aad = torch.mean(torch.acos(cos)).item() / pi
	return aad


def cosine(v1,v2):
	return np.sum(v1*v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2))+1e-10)


def getRMSE(t,v):
	error = np.sum(np.abs(t-v))/np.sum(np.abs(t))
	return np.sqrt(np.mean(error))

def GetMetrics(Vec,args):
	'''
	LERP_PSNR = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-LERP-PSNR.txt','w')
	LERP_AAD = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-LERP-AAD.txt','w')
	LERP_RMSE = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-LERP-RMSE.txt','w')
	TSR_PSNR = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-'+args.mode+'-PSNR.txt','w')
	TSR_AAD = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-'+args.mode+'-AAD.txt','w')
	TSR_RMSE = open('/afs/crc.nd.edu/user/j/jhan5/TSR-VFD/Metrics/'+args.dataset+'-'+args.mode+'-RMSE.txt','w')
	'''
	idx = 0
	psnr_u = 0
	psnr_v = 0
	psnr_w = 0
	psnr = 0
	aad = 0
	rae_u = 0
	rae_v = 0
	rae_w = 0
	rae = 0
	psnr_lerp = 0
	aad_lerp = 0
	rmse_lerp = 0
	for i in range(0,len(Vec.data),args.interval+1):
		print(i)
		if (i+args.interval+1)<len(Vec.data):
			s = Vec.data[i]
			e = Vec.data[i+args.interval+1]
			for j in range(1,args.interval+1):
				gt = Vec.data[i+j]
				tsr = np.fromfile(args.result_path+'TSR/'+args.dataset+'{:03d}'.format(i+j+1)+'.vec',dtype='<f')
				tsr = tsr.reshape(Vec.dim[2],Vec.dim[1],Vec.dim[0],3).transpose()
				linear = j/(args.interval+1)*e+(1-j/(args.interval+1))*s
				LERP = np.asarray(linear,dtype='<f')
				LERP = LERP.flatten('F')
				LERP.tofile(args.result_path+'LERP/'+args.dataset+'{:03d}'.format(i+j+1)+'.vec',format='<f')
				p = getPSNR(gt,tsr)
				#TSR_PSNR.write(str(p))
				#TSR_PSNR.write('\t')
				psnr += p

				p = getPSNR(gt[0],tsr[0])
				psnr_u += p

				p = getPSNR(gt[1],tsr[1])
				psnr_v += p

				p = getPSNR(gt[2],tsr[2])
				psnr_w += p
				p = getPSNR(gt,linear)
				#LERP_PSNR.write(str(p))
				#LERP_PSNR.write('\t')
				psnr_lerp += p
				p = getAAD(gt,tsr)
				#TSR_AAD.write(str(p))
				#TSR_AAD.write('\t')
				aad += p
				p = getAAD(gt,linear)
				#LERP_AAD.write(str(p))
				#LERP_AAD.write('\t')
				aad_lerp += p
				p = getRMSE(gt,tsr)
				#TSR_RMSE.write(str(p))
				#TSR_RMSE.write('\t')
				rae += p

				p = getRMSE(gt[0],tsr[0])
				rae_u += p

				p = getRMSE(gt[1],tsr[1])
				rae_v += p

				p = getRMSE(gt[2],tsr[2])
				rae_w += p
				p = getRMSE(gt,linear)
				#LERP_RMSE.write(str(p))
				#LERP_RMSE.write('\t')
				rmse_lerp += p

				idx += 1
	#LERP_RMSE.close()
	#LERP_AAD.close()
	#LERP_PSNR.close()
	#TSR_RMSE.close()
	#TSR_AAD.close()
	#TSR_PSNR.close()

	print("=====Average PSNR for "+args.mode+"=====")
	print(str(psnr/idx))
	print(str(psnr_u/idx))
	print(str(psnr_v/idx))
	print(str(psnr_w/idx))

	print("=====Average PSNR for LERP=====")
	print(str(psnr_lerp/idx))

	print("=====Average AAD for "+args.mode+"=====")
	print(str(aad/idx))
	print("=====Average AAD for LERP=====")
	print(str(aad_lerp/idx))

	print("=====Average RSME for "+args.mode+"=====")
	print(str(rae/idx))
	print(str(rae_u/idx))
	print(str(rae_v/idx))
	print(str(rae_w/idx))
	print("=====Average RMSE for LERP=====")
	print(str(rmse_lerp/idx))
