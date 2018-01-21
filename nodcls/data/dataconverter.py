import numpy as np
import csv
import pandas as pd
import SimpleITK as sitk
import os
import os.path
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
# read groundtruth from original data space
# remove data of 0 value
pdframe = pd.read_csv('annotationdetclsgt.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
srslst = pdframe['seriesuid'].tolist()[1:]
crdxlst = pdframe['coordX'].tolist()[1:]
crdylst = pdframe['coordY'].tolist()[1:]
crdzlst = pdframe['coordZ'].tolist()[1:]
dimlst = pdframe['diameter_mm'].tolist()[1:]
mlglst = pdframe['malignant'].tolist()[1:]
dct = {}
for idx in xrange(len(srslst)):
    # if mlglst[idx] == '0':
    #     continue
    assert mlglst[idx] in ['1', '0']
    vlu = [float(crdxlst[idx]), float(crdylst[idx]), float(crdzlst[idx]), float(dimlst[idx]), int(mlglst[idx])]
    if srslst[idx] in dct:
        dct[srslst[idx]].append(vlu)
    else:
        dct[srslst[idx]] = [vlu]
# convert it to the preprocessed space
newlst = []
rawpath = '/media/data1/wentao/tianchi/luna16/lunaall/'
preprocesspath = '/media/data1/wentao/tianchi/luna16/preprocess/lunaall/'
resolution = np.array([1,1,1])
def process(pid):
    # print pid
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(rawpath, pid+'.mhd'))
    spacing = np.load(os.path.join(preprocesspath, pid+'_spacing.npy'))
    extendbox = np.load(os.path.join(preprocesspath, pid+'_extendbox.npy'))
    origin = np.load(os.path.join(preprocesspath, pid+'_origin.npy'))
    if isflip:
        Mask = np.load(os.path.join(preprocesspath, pid+'_mask.npy'))
    retlst = []
    for vlu in dct[pid]:
        pos = worldToVoxelCoord(vlu[:3][::-1], origin=origin, spacing=spacing)
        if isflip:
            pos[1:] = Mask.shape[1:3] - pos[1:]
        label = np.concatenate([pos, [vlu[3]/spacing[1]]])
        label2 = np.expand_dims(np.copy(label), 1)
        # print label2.shape
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
        retlst.append([pid, label2[0,0], label2[0,1], label2[0,2], label2[0,3], vlu[-1]])
    return retlst
from multiprocessing import Pool
p = Pool(30)
newlst = p.map(process, dct.keys())
p.close()
print(len(dct.keys()), len(newlst))
# for pid in dct.keys():
#     print pid
#     Mask,origin,spacing,isflip = load_itk_image(os.path.join(rawpath, pid+'.mhd'))
#     spacing = np.load(os.path.join(preprocesspath, pid+'_spacing.npy'))
#     extendbox = np.load(os.path.join(preprocesspath, pid+'_extendbox.npy'))
#     origin = np.load(os.path.join(preprocesspath, pid+'_origin.npy'))
#     if isflip:
#         Mask = np.load(os.path.join(preprocesspath, pid+'_mask.npy'))
#     for vlu in dct[pid]:
#         pos = worldToVoxelCoord(vlu[:3][::-1], origin=origin, spacing=spacing)
#         if isflip:
#             pos[1:] = Mask.shape[1:3] - pos[1:]
#         label = np.concatenate([pos, [vlu[3]/spacing[1]]])
#         label2 = np.expand_dims(np.copy(label), 1)
#         # print label2.shape
#         label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
#         label2[3] = label2[3]*spacing[1]/resolution[1]
#         label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
#         label2 = label2[:4].T
#         newlst.append([pid, label2[0,0], label2[0,1], label2[0,2], label2[0,3], vlu[-1]])
# save it to the csv
savecsv = 'annotationdetclsconv_v3.csv'
fid = open(savecsv, 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
for idx in xrange(len(newlst)):
    for subidx in xrange(len(newlst[idx])):
        writer.writerow(newlst[idx][subidx])
fid.close()