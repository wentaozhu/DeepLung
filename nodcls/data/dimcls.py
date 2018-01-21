import numpy as np
import pandas as pd 
import os
CROPSIZE = 36
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

traindata = np.zeros((len(newlst)-ntest,))
trainlabel = np.zeros((len(newlst)-ntest,))
testdata = np.zeros((ntest,))
testlabel = np.zeros((ntest,))

trainidx = testidx = 0
for idx in xrange(len(newlst)):
	fname = newlst[idx][0]
	if fname.split('-')[0] in testfnamelst:
		testdata[testidx] = newlst[idx][-2]
		testlabel[testidx] = newlst[idx][-1]
		testidx += 1
	else:
		traindata[trainidx] = newlst[idx][-2]
		trainlabel[trainidx] = newlst[idx][-1]
		trainidx += 1

tracclst = []
teacclst = []
thlst = np.sort(traindata).tolist()
besttr = bestte = 0
for th in thlst:
	tracc = np.mean(trainlabel == (traindata > th))
	teacc = np.mean(testlabel == (testdata > th))
	if tracc > besttr:
		besttr = tracc
		bestte = teacc
	tracclst.append(tracc)
	teacclst.append(teacc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.plot(thlst, tracclst, label='train acc')
plt.plot(thlst, teacclst, label='test acc')
plt.xlabel('Threshold for diameter (mm)')
plt.ylabel('Diagnosis (malignant vs. benign) accuracy (%)')
plt.title('Diagnosis accuracy using diameter feature on fold 1')
plt.legend()
plt.savefig('accwrtdim.png')
print max(teacclst)

print besttr, bestte