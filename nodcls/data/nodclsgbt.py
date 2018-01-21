import numpy as np
import pandas as pd 
import mahotas
from mahotas.features.lbp import lbp
CROPSIZE = 17#24#30#36
print CROPSIZE
pdframe =  pd.read_csv('annotationdetclsconv_v3.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
srslst = pdframe['seriesuid'].tolist()[1:]
crdxlst = pdframe['coordX'].tolist()[1:]
crdylst = pdframe['coordY'].tolist()[1:]
crdzlst = pdframe['coordZ'].tolist()[1:]
dimlst = pdframe['diameter_mm'].tolist()[1:]
mlglst = pdframe['malignant'].tolist()[1:]

newlst = []
import csv
fid = open('annotationdetclsconvfnl_v3.csv', 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
for i in xrange(len(srslst)):
	writer.writerow([srslst[i]+'-'+str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
	newlst.append([srslst[i]+'-'+str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
fid.close()

preprocesspath = '/media/data1/wentao/tianchi/luna16/preprocess/lunaall/'
savepath = '/media/data1/wentao/tianchi/luna16/cls/crop_v3/'
import os
import os.path 
# if not os.path.exists(savepath): os.mkdir(savepath)
# for idx in xrange(len(newlst)):
# 	fname = newlst[idx][0]
# 	# if fname != '1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948-581': continue
# 	pid = fname.split('-')[0]
# 	crdx = int(float(newlst[idx][1]))
# 	crdy = int(float(newlst[idx][2]))
# 	crdz = int(float(newlst[idx][3]))
# 	dim = int(float(newlst[idx][4]))
# 	data = np.load(os.path.join(preprocesspath, pid+'_clean.npy'))
# 	bgx = max(0, crdx-CROPSIZE/2)
# 	bgy = max(0, crdy-CROPSIZE/2)
# 	bgz = max(0, crdz-CROPSIZE/2)
# 	cropdata = np.ones((CROPSIZE, CROPSIZE, CROPSIZE))*170
# 	cropdatatmp = np.array(data[0, bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
# 	cropdata[CROPSIZE/2-cropdatatmp.shape[0]/2:CROPSIZE/2-cropdatatmp.shape[0]/2+cropdatatmp.shape[0], \
# 	    CROPSIZE/2-cropdatatmp.shape[1]/2:CROPSIZE/2-cropdatatmp.shape[1]/2+cropdatatmp.shape[1], \
# 	    CROPSIZE/2-cropdatatmp.shape[2]/2:CROPSIZE/2-cropdatatmp.shape[2]/2+cropdatatmp.shape[2]] = np.array(2-cropdatatmp)
# 	assert cropdata.shape[0] == CROPSIZE and cropdata.shape[1] == CROPSIZE and cropdata.shape[2] == CROPSIZE
# 	np.save(os.path.join(savepath, fname+'.npy'), cropdata)

# train use gbt
subset1path = '/media/data1/wentao/tianchi/luna16/subset1/'
testfnamelst = []
for fname in os.listdir(subset1path):
	if fname.endswith('.mhd'):
		testfnamelst.append(fname[:-4])
ntest = 0
for idx in xrange(len(newlst)):
	fname = newlst[idx][0]
	if fname.split('-')[0] in testfnamelst: ntest +=1
print 'ntest', ntest, 'ntrain', len(newlst)-ntest

traindata = np.zeros((len(newlst)-ntest, CROPSIZE*CROPSIZE*CROPSIZE))
trainlabel = np.zeros((len(newlst)-ntest,))
testdata = np.zeros((ntest, CROPSIZE*CROPSIZE*CROPSIZE))
testlabel = np.zeros((ntest,))

trainidx = testidx = 0
for idx in xrange(len(newlst)):
	fname = newlst[idx][0]
	# print fname
	data = np.load(os.path.join(savepath, fname+'.npy'))
	# print data.shape
	bgx = data.shape[0]/2-CROPSIZE/2
	bgy = data.shape[1]/2-CROPSIZE/2
	bgz = data.shape[2]/2-CROPSIZE/2
	data = np.array(data[bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
	if fname.split('-')[0] in testfnamelst:
		testdata[testidx, :] = np.reshape(data, (-1,)) / 255
		# testdata[testidx, -4] = newlst[idx][1]
		# testdata[testidx, -3] = newlst[idx][2]
		# testdata[testidx, -2] = newlst[idx][3]
		# testdata[testidx, -1] = newlst[idx][4]
		testlabel[testidx] = newlst[idx][-1]
		testidx += 1
	else:
		traindata[trainidx, :] = np.reshape(data, (-1,)) / 255
		# traindata[trainidx, -4] = newlst[idx][1]
		# traindata[trainidx, -3] = newlst[idx][2]
		# traindata[trainidx, -2] = newlst[idx][3]
		# traindata[trainidx, -1] = newlst[idx][4]
		trainlabel[trainidx] = newlst[idx][-1]
		trainidx += 1
maxtraindata1 = max(traindata[:, -1])
# traindata[:, -1] = np.array(traindata[:, -1] / maxtraindata1)
# maxtraindata2 = max(traindata[:, -2])
# traindata[:, -2] = np.array(traindata[:, -2] / maxtraindata2)
# maxtraindata3 = max(traindata[:, -3])
# traindata[:, -3] = np.array(traindata[:, -3] / maxtraindata3)
# maxtraindata4 = max(traindata[:, -4])
# traindata[:, -4] = np.array(traindata[:, -4] / maxtraindata4)
# testdata[:, -1] = np.array(testdata[:, -1] / maxtraindata1)
# testdata[:, -2] = np.array(testdata[:, -2] / maxtraindata2)
# testdata[:, -3] = np.array(testdata[:, -3] / maxtraindata3)
# testdata[:, -4] = np.array(testdata[:, -4] / maxtraindata4)
from sklearn.ensemble import GradientBoostingClassifier as gbt
def gbtfunc(dep):
	m = gbt(max_depth=dep, random_state=0)
	m.fit(traindata, trainlabel)
	predtrain = m.predict(traindata)
	predtest = m.predict_proba(testdata)
	# print predtest.shape, predtest[1,:]
	return np.sum(predtrain == trainlabel) / float(traindata.shape[0]), \
	    np.mean((predtest[:,1]>0.5).astype(int) == testlabel), predtest # / float(testdata.shape[0]),
# trainacc, testacc, predtest = gbtfunc(3)
# print trainacc, testacc
# np.save('pixradiustest.npy', predtest[:,1])
from multiprocessing import Pool
p = Pool(30)
acclst = p.map(gbtfunc, range(1,9,1))#3,4,1))#5,1))#1,9,1))
for acc in acclst:
	print("{0:.4f}".format(acc[0]), "{0:.4f}".format(acc[1]))
p.close()
# for dep in xrange(1,9,1):
# 	m = gbt(max_depth=dep)
# 	m.fit(traindata, trainlabel)
# 	print dep, 'trainacc', np.sum(m.predict(traindata) == trainlabel) / float(traindata.shape[0])
# 	print dep, 'testacc', np.sum(m.predict(testdata) == testlabel) / float(testdata.shape[0])