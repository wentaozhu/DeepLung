import pandas as pd 
import SimpleITK as sitk
import os
import os.path
import numpy as np
fold = 1
def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))    
    return numpyImage, numpyOrigin, numpySpacing,isflip
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord
# read map file
mapfname = 'LIDC-IDRI-mappingLUNA16'
sidmap = {}
fid = open(mapfname, 'r')
line = fid.readline()
line = fid.readline()
while line:
	pidlist = line.split(' ')
	# print pidlist
	pid = pidlist[0] 
	stdid = pidlist[1] 
	srsid = pidlist[2]
	if srsid not in sidmap:
		sidmap[srsid] = [pid, stdid]
	else:
		assert sidmap[srsid][0] == pid
		assert sidmap[srsid][1] == stdid
	line = fid.readline()
fid.close()
# read luna16 annotation
colname = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
lunaantframe = pd.read_csv('annotations.csv', names=colname)
srslist = lunaantframe.seriesuid.tolist()[1:]
cdxlist = lunaantframe.coordX.tolist()[1:]
cdylist = lunaantframe.coordY.tolist()[1:]
cdzlist = lunaantframe.coordZ.tolist()[1:]
dimlist = lunaantframe.diameter_mm.tolist()[1:]
lunaantdict = {}
for idx in xrange(len(srslist)):
	vlu = [float(cdxlist[idx]), float(cdylist[idx]), float(cdzlist[idx]), float(dimlist[idx])]
	if srslist[idx] in lunaantdict:
		lunaantdict[srslist[idx]].append(vlu)
	else:
		lunaantdict[srslist[idx]] = [vlu]
# # convert luna16 annotation to LIDC-IDRI annotation space
# from multiprocessing import Pool
# lunantdictlidc = {}
# for fold in xrange(10):
# 	mhdpath = '/media/data1/wentao/tianchi/luna16/subset'+str(fold)
# 	print 'fold', fold
# 	def getvoxcrd(fname):
# 		sliceim,origin,spacing,isflip = load_itk_image(os.path.join(mhdpath, fname))
# 		lunantdictlidc[fname[:-4]] = []
# 		voxcrdlist = []
# 		for lunaant in lunaantdict[fname[:-4]]:
# 			voxcrd = worldToVoxelCoord(lunaant[:3][::-1], origin, spacing)
# 			voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
# 			voxcrdlist.append(voxcrd)
# 		return voxcrdlist
# 	p = Pool(30)
# 	fnamelist = []
# 	for fname in os.listdir(mhdpath):
# 		if fname.endswith('.mhd') and fname[:-4] in lunaantdict:
# 			fnamelist.append(fname)
# 	voxcrdlist = p.map(getvoxcrd, fnamelist)
# 	listidx = 0
# 	for fname in os.listdir(mhdpath):
# 		if fname.endswith('.mhd') and fname[:-4] in lunaantdict:
# 			lunantdictlidc[fname[:-4]] = []
# 			for subidx, lunaant in enumerate(lunaantdict[fname[:-4]]):
# 				# voxcrd = worldToVoxelCoord(lunaant[:3][::-1], origin, spacing)
# 				# voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
# 				lunantdictlidc[fname[:-4]].append([lunaant, voxcrdlist[listidx][subidx]])
# 			listidx += 1
# 	p.close()
# np.save('lunaantdictlidc.npy', lunantdictlidc)
# read LIDC dataset
lunantdictlidc = np.load('lunaantdictlidc.npy').item()
import xlrd
lidccsvfname = '/media/data1/wentao/LIDC-IDRI/list3.2.xls'
antdict = {}
wb = xlrd.open_workbook(os.path.join(lidccsvfname))
for s in wb.sheets():
	if s.name == 'list3.2':
		for row in range(1, s.nrows):
			valuelist = [int(s.cell(row, 2).value), s.cell(row, 3).value, s.cell(row, 4).value, \
			    int(s.cell(row, 5).value), int(s.cell(row, 6).value), int(s.cell(row, 7).value)]
			assert abs(s.cell(row, 1).value - int(s.cell(row, 1).value)) < 1e-8
			assert abs(s.cell(row, 2).value - int(s.cell(row, 2).value)) < 1e-8
			assert abs(s.cell(row, 5).value - int(s.cell(row, 5).value)) < 1e-8
			assert abs(s.cell(row, 6).value - int(s.cell(row, 6).value)) < 1e-8 
			assert abs(s.cell(row, 7).value - int(s.cell(row, 7).value)) < 1e-8
			for col in range(9, 16):
				if s.cell(row, col).value != '':
					if isinstance(s.cell(row, col).value, float):
						valuelist.append(str(int(s.cell(row, col).value)))
						assert abs(s.cell(row, col).value - int(s.cell(row, col).value)) < 1e-8
					else:
						valuelist.append(s.cell(row, col).value)
			if s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value)) not in antdict:
				antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))] = [valuelist]
			else:
				antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))].append(valuelist)
# update LIDC annotation with series number, rather than scan id
import dicom
LIDCpath = '/media/data1/wentao/LIDC-IDRI/DOI/'
antdictscan = {}
for k, v in antdict.iteritems():
	pid, scan = k.split('_')
	hasscan = False
	for sdu in os.listdir(os.path.join(LIDCpath, 'LIDC-IDRI-'+pid)):
		for srs in os.listdir(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu])):
			if srs.endswith('.npy'):
				print 'npy', pid, scan, srs
				continue
			RefDs = dicom.read_file(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu, srs, '000006.dcm']))
			# print scan, str(RefDs[0x20, 0x11].value)
			if str(RefDs[0x20, 0x11].value) == scan or scan == '0': 
				if hasscan: print 'rep', pid, sdu, srs
				hasscan = True
				antdictscan[pid+'_'+srs] = v
				break
	if not hasscan: print 'not found', pid, scan, sdu, srs
# find the match from LIDC-IDRI annotation
import math
lunaantdictnodid = {}
maxdist = 0
for srcid, lunaantlidc in lunantdictlidc.iteritems():
	lunaantdictnodid[srcid] = []
	pid, stdid = sidmap[srcid]
	# print pid
	pid = pid[len('LIDC-IDRI-'):]
	for lunantdictlidcsub in lunaantlidc:
		lunaant = lunantdictlidcsub[0]
		voxcrd = lunantdictlidcsub[1] # z y x
		mindist, minidx = 1e8, -1
		if srcid in ['1.3.6.1.4.1.14519.5.2.1.6279.6001.174692377730646477496286081479', '1.3.6.1.4.1.14519.5.2.1.6279.6001.300246184547502297539521283806']:
			continue
		for idx, lidcant in enumerate(antdictscan[pid+'_'+srcid]):
			dist = math.pow(voxcrd[0] - lidcant[3], 2) # z
			dist = math.pow(voxcrd[1] - lidcant[4], 2) # y
			dist += math.pow(voxcrd[2] - lidcant[5], 2) # x
			if dist < mindist:
				mindist = dist
				minidx = idx
		if mindist > 71:#15.1:
			print srcid, pid, voxcrd, antdictscan[pid+'_'+srcid], mindist
		maxdist = max(maxdist, mindist)
		lunaantdictnodid[srcid].append([lunaant, antdictscan[pid+'_'+srcid][minidx][6:]])
# np.save('lunaantdictnodid.npy', lunaantdictnodid)
print 'maxdist', maxdist
# save it into a csv
# import csv
# savename = 'annotationnodid.csv'
# fid = open(savename, 'w')
# writer = csv.writer(fid)
# writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])
# for srcid, ant in lunaantdictnodid.iteritems():
# 	for antsub in ant:
# 		writer.writerow([srcid] + [antsub[0][0], antsub[0][1], antsub[0][2], antsub[0][3]] + antsub[1])
# fid.close()
# fd 1
fd1lst = []
for fname in os.listdir('/media/data1/wentao/tianchi/luna16/subset'+str(fold)+'/'):
	if fname.endswith('.mhd'): fd1lst.append(fname[:-4])
# find the malignancy, shape information from xml file
import xml.dom.minidom
ndoc = 0
lunadctclssgmdict = {}
mallstall, callstall, sphlstall, marlstall, loblstall, spilstall, texlstall = [], [], [], [], [], [], []
for srsid, extant in lunaantdictnodid.iteritems():
	if srsid not in fd1lst: continue
	lunadctclssgmdict[srsid] = []
	pid, stdid = sidmap[srsid]
	for extantvlu in extant:
		getnodid = []
		nant = 0
		mallst = []
		for fname in os.listdir(os.path.join(*['/media/data1/wentao/LIDC-IDRI/DOI/', pid, stdid, srsid])):
			if fname.endswith('.xml'):
				nant += 1
				dom = xml.dom.minidom.parse(os.path.join(*['/media/data1/wentao/LIDC-IDRI/DOI/', pid, stdid, srsid, fname]))
				root = dom.documentElement
				rsessions = root.getElementsByTagName('readingSession')
				for rsess in rsessions:
					unblinds = rsess.getElementsByTagName('unblindedReadNodule')
					for unb in unblinds:
						nod = unb.getElementsByTagName('noduleID')
						if len(nod) != 1: 
							print 'more nod', nod
							continue
						if nod[0].firstChild.data in extantvlu[1]:
							getnodid.append(nod[0].firstChild.data)
							mal = unb.getElementsByTagName('malignancy')
							if len(mal) == 1 and int(mal[0].firstChild.data) in range(1, 6, 1):
								mallst.append(float(mal[0].firstChild.data))
		# print(getnodid, extantvlu[1], nant)
		if len(getnodid) > len(extantvlu[1]): 
			print pid, srsid
			# assert 1 == 0
		ndoc = max(ndoc, len(getnodid), len(extantvlu[1]))
		vlulst = [srsid, extantvlu[0][0], extantvlu[0][1], extantvlu[0][2], extantvlu[0][3]]
		if len(mallst) == 0: vlulst.append(0)
		else: vlulst.append(sum(mallst)/float(len(mallst)))
		lunadctclssgmdict[srsid].append(vlulst+mallst)
import csv
# load predition array
pixdimpred = np.load('../../../CTnoddetector/training/nodcls/besttestpred.npy')#'pixradiustest.npy')
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
subset1path = '/media/data1/wentao/tianchi/luna16/subset'+str(fold)+'/'
testfnamelst = []
for fname in os.listdir(subset1path):
	if fname.endswith('.mhd'):
		testfnamelst.append(fname[:-4])
ntest = 0
for idx in xrange(len(newlst)):
	fname = newlst[idx][0]
	if fname.split('-')[0] in testfnamelst: ntest +=1
print 'ntest', ntest, 'ntrain', len(newlst)-ntest
prednamelst = {}
predacc = 0
predidx = 0
# predlabellst = []
for idx in xrange(len(newlst)):
	fname = newlst[idx][0]
	if fname.split('-')[0] in testfnamelst:
		# print newlst[idx][-1], pixdimpred[predidx]
		if int(pixdimpred[predidx]>0.5) == int(newlst[idx][-1]): predacc += 1
		if fname.split('-')[0] not in prednamelst: 
			prednamelst[fname.split('-')[0]] = [[pixdimpred[predidx], fname.split('-')[1]]]
		else:
			prednamelst[fname.split('-')[0]].append([pixdimpred[predidx], fname.split('-')[1]])
		predidx += 1
print 'pred acc', predacc/float(predidx)
pixdimidx = -1
# savename = 'annotationdetclssgm_doctor_fd2.csv'
# fid = open(savename, 'w')
# writer = csv.writer(fid)
# writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
doctornacc, doctornpid = [0]*ndoc, [0]*ndoc
nacc = 0
ntot = 0
for srsid, extant in lunadctclssgmdict.iteritems():
	curidx = 0
	if srsid not in fd1lst: continue
	for subextant in extant:
		if subextant[5] in [3, 0]: continue
		if abs(subextant[5] - 3) < 1e-2: continue
		if subextant[5] > 3: subextant[5] = 1
		else: subextant[5] = 0
		if subextant[5] == int(prednamelst[srsid][curidx][0]>0.5): nacc += 1
		ntot += 1
		# writer.writerow(subextant)
		for did in xrange(6, len(subextant), 1):
			# if 0.499 <= prednamelst[srsid][curidx] <= 0.501: continue
			if subextant[did] == 3: continue
			if subextant[5] != (prednamelst[srsid][curidx][0]>0.5):
				print(srsid+'-'+prednamelst[srsid][curidx][1], prednamelst[srsid][curidx][0], subextant[5]) 
			# if subextant[5] == 1 and prednamelst[srsid][curidx][0]>0.5:# subextant[did] > 3:  # we treat 3 as the positive label
			# 	if subextant[did] < 3: print(srsid+'-'+prednamelst[srsid][curidx][1], prednamelst[srsid][curidx][0], did-6)
			# 	doctornacc[did-6] += 1
			# elif subextant[5] == 0 and prednamelst[srsid][curidx][0]<=0.5:# subextant[did] < 3:
			# 	if subextant[did] > 3: print(srsid+'-'+prednamelst[srsid][curidx][1], prednamelst[srsid][curidx][0], did-6)
			# 	doctornacc[did-6] += 1
			# if subextant[did] != 3:
			# 	doctornpid[did-6] += 1
		curidx += 1
fid.close()
print(nacc / float(ntot)) 	
for i in xrange(ndoc):
	print(i, doctornacc[i], doctornpid[i], doctornacc[i]/float(doctornpid[i]))