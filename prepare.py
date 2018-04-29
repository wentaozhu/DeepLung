import os
import shutil
import numpy as np
import sys
# sys.path.append('./')
# import config_training
from config_training import config 
# config = config_training.config
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial

import warnings

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

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

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw

def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python_tianchi(case_path):
    # case = load_scan(case_path)
    # case_pixels, spacing = get_pixels_hu(case)
    ''' For the mhd file reader '''
    resolution = np.array([1,1,1])
    sliceim,origin,spacing,isflip = load_itk_image(case_path+'.mhd')
    if isflip:
        sliceim = sliceim[:,::-1,::-1]
        print('flip!')
    # sliceim = lumTrans(sliceim)
    # sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    case_pixels = np.array(sliceim)

    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing, origin, isflip

def savenpy(id, annos, filelist, data_path, prep_folder):    
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    im, m1, m2, spacing, origin, isflip = step1_python_tianchi(os.path.join(data_path, name))
    missingmask = False
    if os.path.exists(os.path.join(prep_folder,name+'_clean.npy')) and \
        os.path.exists(os.path.join(prep_folder,name+'_originbox.npy')) and \
        os.path.exists(os.path.join(prep_folder,name+'_spacing.npy')) and \
        os.path.exists(os.path.join(prep_folder,name+'_origin.npy')) and \
        os.path.exists(os.path.join(prep_folder,name+'_label.npy')):
        if not isflip:
            print 'skip', name
            return
        else:
            missingmask = True

    print 'process', name
    label = annos[annos[:,0]==name]
    # label = label.astype('float')
    label = label[:, [3,1,2,4]].astype('float') # z, y, x, d
    
    
    Mask = m1 + m2
    
    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    xx,yy,zz = np.where(Mask)
    if xx.size == 0 or yy.size == 0 or zz.size == 0:
        print name 
        assert 1 == 0

    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')
    if extendbox[0,0] == extendbox[0,1] or extendbox[1,0] == extendbox[1,1] or extendbox[2,0] == extendbox[2,1]:
        print name
        assert 1==0

    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    if missingmask:
        np.save(os.path.join(prep_folder,name+'_mask.npy'), Mask)
        print 'skip', name
        return
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'), sliceim)
    np.save(os.path.join(prep_folder,name+'_originbox.npy'), extendbox)
    np.save(os.path.join(prep_folder,name+'_spacing.npy'), spacing)
    np.save(os.path.join(prep_folder,name+'_origin.npy'), origin)
    print im.shape, '_clean', sliceim.shape, '_originbox', extendbox.shape, '_space', spacing, '_origin', origin

    this_annos = np.copy(annos[annos[:,0]==name])
    label = []
    print 'label', this_annos.shape, name
    if len(this_annos)>0:
        
        for c in this_annos:
            pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
            if isflip:
                pos[1:] = Mask.shape[1:3]-pos[1:]
            label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
        
    label = np.array(label)
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    else:
        label2 = np.copy(label).T
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)
    print name

def full_prep(train=True, val=True, test=True):
    warnings.filterwarnings("ignore")
    #preprocess_result_path = './prep_result'
    train_prep_folder = config['train_preprocess_result_path']
    val_prep_folder = config['val_preprocess_result_path']
    test_prep_folder = config['test_preprocess_result_path']
    
    train_data_path = config['train_data_path']
    val_data_path = config['val_data_path']
    test_data_path = config['test_data_path']
    
    finished_flag = '.flag_preptianchi'
    
    if not os.path.exists(finished_flag):
        trainlabelfiles = config['train_annos_path']
        vallabelfiles = config['val_annos_path']
        testlabelfiles = config['test_annos_path']

        traincontent = np.array(pandas.read_csv(trainlabelfiles))
        traincontent = traincontent[traincontent[:, 0] != np.nan]
        trainalllabel = traincontent[1:, :] # filename, x, y, z, d
        trainfilelist = []
        for f in os.listdir(config['train_data_path']):
            if f.endswith('.mhd'):
                if f[:-4] in config['black_list']:
                    continue
                trainfilelist.append(f[:-4])

        valcontent = np.array(pandas.read_csv(vallabelfiles))
        valcontent = valcontent[valcontent[:, 0] != np.nan]
        valalllabel = valcontent[1:, :] # filename, x, y, z, d
        valfilelist = []
        for f in os.listdir(config['val_data_path']):
            if f.endswith('.mhd'):
                if f[:-4] in config['black_list']:
                    continue
                valfilelist.append(f[:-4])

        testcontent = np.array(pandas.read_csv(testlabelfiles))
        testcontent = testcontent[testcontent[:, 0] != np.nan]
        testalllabel = testcontent[1:, :] # filename, x, y, z, d
        testfilelist = []
        for f in os.listdir(config['test_data_path']):
            if f.endswith('.mhd'):
                if f[:-4] in config['black_list']:
                    continue
                testfilelist.append(f[:-4])

        if not os.path.exists(train_prep_folder):
            os.mkdir(train_prep_folder)
        if not os.path.exists(val_prep_folder):
            os.mkdir(val_prep_folder)
        if not os.path.exists(test_prep_folder):
            os.mkdir(test_prep_folder)
        #eng.addpath('preprocessing/',nargout=0)
        if train:
            print('starting train preprocessing')
            pool = Pool(10)
            partial_savenpy = partial(savenpy, annos=trainalllabel, filelist=trainfilelist, data_path=train_data_path, prep_folder=train_prep_folder)
            N = len(trainfilelist)
            savenpy(1)
            _=pool.map(partial_savenpy, range(N))
            print('end train preprocessing')
        if val:
            print('starting val preprocessing')
            partial_savenpy = partial(savenpy, annos=valalllabel, filelist=valfilelist, data_path=val_data_path, prep_folder=val_prep_folder)
            N = len(valfilelist)
                savenpy(1)
            _=pool.map(partial_savenpy, range(N))
            print('end val preprocessing')
        if test:
            print('starting test preprocessing')
            partial_savenpy = partial(savenpy, annos=testalllabel, filelist=testfilelist, data_path=test_data_path, prep_folder=test_prep_folder)
            N = len(testfilelist)
                savenpy(1)
            _=pool.map(partial_savenpy, range(N))
            pool.close()
            pool.join()
            print('end test preprocessing')
    f= open(finished_flag,"w+")        
    
def splitvaltestcsv():
    testfiles = []
    for f in os.listdir(config['test_data_path']):
        if f.endswith('.mhd'):
            testfiles.append(f[:-4])
    valcsvlines = []
    testcsvlines = []
    import csv 
    valf = open(config['val_annos_path'], 'r')
    valfcsv = csv.reader(valf)
    for line in valfcsv:
        if line[0] in testfiles:
            testcsvlines.append(line)
        else:
            valcsvlines.append(line)
    valf.close()
    testf = open(config['test_annos_path']+'annotations.csv', 'w')
    testfcsv = csv.writer(testf)
    for line in testcsvlines:
        testfcsv.writerow(line)
    testf.close()
    valf = open(config['val_annos_path'], 'w')
    valfcsv = csv.writer(valf)
    for line in valcsvlines:
        valfcsv.writerow(line)
    valf.close()

def savenpy_luna(id, annos, filelist, luna_segment, luna_data,savepath):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    
    sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))

    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    this_annos = np.copy(annos[annos[:,0]==(name)])        

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2

        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath, name+'_clean.npy'), sliceim)
        np.save(os.path.join(savepath, name+'_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name+'_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name+'_origin.npy'), origin)
        np.save(os.path.join(savepath, name+'_mask.npy'), Mask)

    if islabel:
        this_annos = np.copy(annos[annos[:,0]==(name)])
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'), label2)
        
    print(name)

def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        annos = np.array(pandas.read_csv(luna_label))
        pool = Pool()
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for setidx in xrange(10):
            print 'process subset', setidx
            filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data+'subset'+str(setidx)) if f.endswith('.mhd') ]
            if not os.path.exists(savepath+'subset'+str(setidx)):
                os.mkdir(savepath+'subset'+str(setidx))
            partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=luna_data+'subset'+str(setidx)+'/', 
                                       savepath=savepath+'subset'+str(setidx)+'/')
            N = len(filelist)
            #savenpy(1)
            _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")


if __name__=='__main__':
    preprocess_luna()
