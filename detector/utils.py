import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

def getFreeId():
    import pynvml 

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def split4(data,  max_stride, margin):
    splits = []
    data = torch.Tensor.numpy(data)
    _,c, z, h, w = data.shape

    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    pad = int(np.ceil(float(z)/max_stride)*max_stride)-z
    leftpad = pad/2
    pad = [[0,0],[0,0],[leftpad,pad-leftpad],[0,0],[0,0]]
    data = np.pad(data,pad,'constant',constant_values=-1)
    data = torch.from_numpy(data)
    splits.append(data[:, :, :, :h_width, :w_width])
    splits.append(data[:, :, :, :h_width, -w_width:])
    splits.append(data[:, :, :, -h_width:, :w_width])
    splits.append(data[:, :, :, -h_width:, -w_width:])
    
    return torch.cat(splits, 0)

def combine4(output, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        splits[0].shape[0],
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    h0 = output.shape[1] / 2
    h1 = output.shape[1] - h0
    w0 = output.shape[2] / 2
    w1 = output.shape[2] - w0

    splits[0] = splits[0][:, :h0, :w0, :, :]
    output[:, :h0, :w0, :, :] = splits[0]

    splits[1] = splits[1][:, :h0, -w1:, :, :]
    output[:, :h0, -w1:, :, :] = splits[1]

    splits[2] = splits[2][:, -h1:, :w0, :, :]
    output[:, -h1:, :w0, :, :] = splits[2]

    splits[3] = splits[3][:, -h1:, -w1:, :, :]
    output[:, -h1:, -w1:, :, :] = splits[3]

    return output

def split8(data,  max_stride, margin):
    splits = []
    if isinstance(data, np.ndarray):
        c, z, h, w = data.shape
    else:
        _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                if isinstance(data, np.ndarray):
                    splits.append(data[np.newaxis, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
                else:
                    splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

                
    if isinstance(data, np.ndarray):
        return np.concatenate(splits, 0)
    else:
        return torch.cat(splits, 0)

    

def combine8(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 2
    h_width = h / 2
    w_width = w / 2
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output


def split16(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine16(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 4
    h_width = h / 2
    w_width = w / 2
    splitzstart = splits[0].shape[0]/2-z_width/2
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                      [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output

def split32(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine32(splits, z, h, w):
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 2))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output



def split64(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine64(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 4))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splitzstart = splits[0].shape[0]/2-z_width/2
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                          [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output

def plotlog(logfile, savepath):
    traintpr = []
    traintnr = []
    trainloss = []
    trainclassifyloss = []
    trainregresslossx = []
    trainregresslossy = []
    trainregresslossz = []
    trainregresslossd = []

    valtpr = []
    valtnr = []
    valloss = []
    valclassifyloss = []
    valregresslossx = []
    valregresslossy = []
    valregresslossz = []
    valregresslossd = []

    eps = 1
    f = open(logfile, 'r')
    line = f.readline()
    while line:
        if line.startswith('Epoch '+'{:03d}'.format(eps)+' (lr '):
            trainline1 = f.readline()
            strlist = trainline1.split('Train:      tpr ')
            # print strlist
            strlist1 = strlist[1].split(',')
            # print strlist1, float(strlist1[0])
            traintpr.append(float(strlist1[0]))
            strlist2 = strlist1[1].split('tnr ')
            # print strlist2, float(strlist2[1])
            traintnr.append(float(strlist2[1])) 

            trainline2 = f.readline()[5:]
            strlist = trainline2.split(', classify loss ')
            # print strlist, float(strlist[0])
            trainloss.append(float(strlist[0]))
            strlist1 = strlist[1].split(', regress loss ')
            # print strlist1, float(strlist1[0])
            trainclassifyloss.append(float(strlist1[0]))
            strlist2 = strlist1[1].split(', ')
            # print strlist2, float(strlist2[0]), float(strlist2[1]), float(strlist2[2]), float(strlist2[3])
            trainregresslossx.append(float(strlist2[0]))
            trainregresslossy.append(float(strlist2[1]))
            trainregresslossz.append(float(strlist2[2]))
            trainregresslossd.append(float(strlist2[3]))

            f.readline()

            valline1 = f.readline()
            strlist = valline1.split('Validation: tpr ')
            # print strlist
            strlist1 = strlist[1].split(',')
            # print strlist1, float(strlist1[0])
            valtpr.append(float(strlist1[0]))
            strlist2 = strlist1[1].split('tnr ')
            # print strlist2, float(strlist2[1])
            valtnr.append(float(strlist2[1])) 

            valline2 = f.readline()[5:]
            strlist = valline2.split(', classify loss ')
            # print strlist, float(strlist[0])
            valloss.append(float(strlist[0]))
            strlist1 = strlist[1].split(', regress loss ')
            # print strlist1, float(strlist1[0])
            valclassifyloss.append(float(strlist1[0]))
            strlist2 = strlist1[1].split(', ')
            # print strlist2, float(strlist2[0]), float(strlist2[1]), float(strlist2[2]), float(strlist2[3])
            valregresslossx.append(float(strlist2[0]))
            valregresslossy.append(float(strlist2[1]))
            valregresslossz.append(float(strlist2[2]))
            valregresslossd.append(float(strlist2[3]))

            eps += 1
        line = f.readline()
    f.close()

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), traintpr, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valtpr, label='val')
    plt.legend()
    plt.title('True Positive Rate')
    plt.savefig(savepath+'tpr.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintnr)+1, 1), traintnr, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valtnr, label='val')
    plt.legend()
    plt.title('True Negative Rate')
    plt.savefig(savepath+'tnr.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainloss, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valloss, label='val')
    plt.legend()
    plt.title('Loss')
    plt.savefig(savepath+'loss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainclassifyloss, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valclassifyloss, label='val')
    plt.legend()
    plt.title('Classification Loss')
    plt.savefig(savepath+'classificationloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossx, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossx, label='val')
    plt.legend()
    plt.title('Regresion X Loss')
    plt.savefig(savepath+'regressionxloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossy, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossy, label='val')
    plt.legend()
    plt.title('Regresion Y Loss')
    plt.savefig(savepath+'regressionyloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossz, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossz, label='val')
    plt.legend()
    plt.title('Regresion Z Loss')
    plt.savefig(savepath+'regressionzloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossd, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossd, label='val')
    plt.legend()
    plt.title('Regresion D Loss')
    plt.savefig(savepath+'regressiondloss.png')

def plotnoduledist(annopath):
    import pandas as pd 
    df = pd.read_csv(annopath+'train/annotations.csv')
    diameter = df['diameter_mm'].reshape((-1,1))

    df = pd.read_csv(annopath+'val/annotations.csv')
    diameter = np.vstack([df['diameter_mm'].reshape((-1,1)), diameter])

    df = pd.read_csv(annopath+'test/annotations.csv')
    diameter = np.vstack([df['diameter_mm'].reshape((-1,1)), diameter])
    fig = plt.figure()
    plt.hist(diameter, normed=True, bins=50)
    plt.ylabel('probability')
    plt.xlabel('Diameters')
    plt.title('Nodule Diameters Histogram')
    plt.savefig('nodulediamhist.png')

def plotnoduledistkaggle(annopath):
    import pandas as pd 
    df = pd.read_csv(annopath)
    diameter = df['diameter']

    dlist = []
    for i in xrange(diameter.shape[0]):
        if diameter[i] > 0:
            dlist.append(diameter[i])
    plt.hist(dlist, normed=True, bins=50)
    plt.ylabel('probability')
    plt.xlabel('Diameters')
    plt.title('Nodule Diameters Histogram')
    plt.savefig('nodulediamhistkaggle.png')

def calrecall(csvpath):
    import pandas as pd
    df = pd.read_csv(csvpath)
    seriesid = df['seriesuid']
    ttol = seriesid.shape[0]
    pos = len(set(seriesid))
    print(pos/float(ttol))

def plothistdiameter(trainpath='/media/data1/wentao/tianchi/preprocessing/newtrain/', 
                     testpath='/media/data1/wentao/tianchi/preprocessing/newtest/'):
    diameterlist = []
    for fname in os.listdir(trainpath):
        if fname.endswith('_label.npy'):
            label = np.load(trainpath+fname)
            for lidx in xrange(label.shape[0]):
                diameterlist.append(label[lidx, -1])
    for fname in os.listdir(testpath):
        if fname.endswith('_label.npy'):
            label = np.load(testpath+fname)
            for lidx in xrange(label.shape[0]):
                diameterlist.append(label[lidx, -1])
    fig = plt.figure()
    plt.hist(diameterlist, 50)
    plt.xlabel('Nodule Diameter')
    plt.ylabel('# Nodules')
    plt.title('Nodule Size Histogram')
    plt.savefig('processnodulesizehist.png')

def getrecall(path='/media/data1/wentao/tianchi/CTnoddetector/training/detector/results/res18/baselinebboxlranchorftbig/test55test/'):
    thresh = np.linspace(-15, 0, 31)
    p = 0
    print thresh.shape, thresh
    # print recall.size
    tot = 0
    for fname in os.listdir(path):
        if fname.endswith('_lbb.npy'):
            tot += 1
    recall = np.zeros((thresh.shape[0], tot))
    tot = 0
    for fname in os.listdir(path):
        if fname.endswith('_lbb.npy'):
            print fname
            label = np.load(path+fname, 'r')
            p += label.shape[0]
            pbb = np.load(path+fname[:-8]+'_pbb.npy', 'r')
            if label.shape[0] == 0: continue
            for thi in xrange(thresh.shape[0]-1, -1, -1):
                if thi != thresh.shape[0]-1 and recall[thi+1, tot] == label.shape[0]:
                    recall[thi, tot] = label.shape[0]
                    continue 
                th = thresh[thi]
                tp = 0
                pbbremain = np.array(pbb[pbb[:,0]>th])
                lidxbegin = 0
                for pidx in xrange(pbbremain.shape[0]):
                    for lidx in xrange(lidxbegin, label.shape[0], 1):
                        if np.sum(np.square(pbbremain[pidx, 1:4] - label[lidx, :3])) <= (label[lidx,-1]/2)**2:
                            tp += 1
                            lidxbegin += 1
                            if lidxbegin == label.shape[0]:
                                break
                            continue
                    if lidxbegin == label.shape[0]:
                        break
                recall[thi, tot] += tp
                # print th, tp
            tot += 1
    recall /= p 
    print 'p', p
    print recall.sum(axis=1)
    fig = plt.figure()
    plt.plot(thresh, recall.sum(axis=1))
    plt.xlabel('Threshold (before sigmoid)')
    plt.ylabel('Recall')
    plt.title('Recall w.r.t. Threshold')
    plt.savefig('recallthresh.png')

def getrecallwrtdetp(path='/media/data1/wentao/tianchi/CTnoddetector/training/detector/results/res18/baselinebboxlranchorftbig96/test64/'):
    thresh = np.linspace(-7, 0, 8)
    import pandas as pd
    import math
    gtcolname = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
    annogt = pd.read_csv('/media/data1/wentao/tianchi/csv/newtest/annotations.csv', names=gtcolname)
    seriesuidgt = annogt.seriesuid.tolist()
    coordXgt = annogt.coordX.tolist()
    coordYgt = annogt.coordY.tolist()
    coordZgt = annogt.coordZ.tolist()
    diameter_mmgt = annogt.diameter_mm.tolist()
    p = 0
    # print recall.size
    tot = 0
    for fname in os.listdir(path):
        if fname.endswith('_lbb.npy'):
            tot += 1
            label = np.load(path+fname, 'r')
            p += label.shape[0]
    print thresh.shape, thresh, tot, p, seriesuidgt[0]
    recall = np.zeros((thresh.shape[0],))
    colnames = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
    for thi in xrange(thresh.shape[0]):
        df = pd.read_csv(path+'predanno'+'{:02d}'.format(int(thresh[thi]))+'.csv', names=colnames)
        seriesuid = df.seriesuid.tolist()
        coordX = df.coordX.tolist()
        # print type(coordX)
        coordY = df.coordY.tolist()
        coordZ = df.coordZ.tolist()
        # print coordX, seriesuid, coordY, coordZ
        for fname in set(seriesuidgt[1:]):
            labelx, labely, labelz, labelr = [], [], [], []
            for seriesidx in xrange(len(seriesuidgt)):
                if seriesuidgt[seriesidx] == fname:
                    labelx.append(float(coordXgt[seriesidx]))
                    labely.append(float(coordYgt[seriesidx]))
                    labelz.append(float(coordZgt[seriesidx]))
                    labelr.append(float(diameter_mmgt[seriesidx]))

            labelsignlist = [0]*len(labelx)
            coordXremain, coordYremain, coordZremain = [], [], []
            for csvidx in xrange(len(seriesuid)):
                # print seriesuid[csvidx], fname[:-8]
                if seriesuid[csvidx] == fname:
                    coordXremain.append(coordX[csvidx])
                    coordYremain.append(coordY[csvidx])
                    coordZremain.append(coordZ[csvidx])
            # print len(coordX), len(coordXremain)#, labelx#, coordXremain
            for coordi in xrange(len(coordXremain)):
                for labeli in xrange(len(label)):
                    if labelsignlist[labeli] == 1:
                        continue
                    try:
                        float(coordXremain[coordi])
                        float(coordYremain[coordi])
                        float(coordZremain[coordi])
                    except:
                        print coordXremain[coordi], coordYremain[coordi], coordZremain[coordi]
                    dist = math.pow(float(coordXremain[coordi]) - labelx[labeli], 2.0)
                    dist += math.pow(float(coordYremain[coordi]) - labely[labeli], 2.0)
                    dist += math.pow(float(coordZremain[coordi]) - labelz[labeli], 2.0)
                    if dist < math.pow(labelr[labeli]/2.0, 2.0):
                        recall[thi] += 1
                        labelsignlist[labeli] = 1
                        # break
                if sum(labelsignlist) == label.shape[0]:
                    break
    recall /= float(p)
    print recall

    fig = plt.figure()
    plt.plot(thresh, recall)
    plt.xlabel('Threshold (before sigmoid)')
    plt.ylabel('Recall')
    plt.title('Recall w.r.t. Threshold (after nms)')
    plt.savefig('recallthreshafternms.png')

if __name__ == '__main__':
    # path = '/media/data1/wentao/CTnoddetector/training/detector/results/res18/ft96/'
    # plotlog(path+'log', savepath=path)
    plotlog('log', savepath='./')
    # plotnoduledist('/mnt/media/wentao/tianchi/csv/')
    # plotnoduledistkaggle('/mnt/media/wentao/tianchi/DSB2017/training/detector/labels/label_job0_full.csv')

    # calrecall('/mnt/media/wentao/tianchi/csv/val/annotations.csv')
    # calrecall('/mnt/media/wentao/tianchi/csv/test/annotations.csv')

    # plothistdiameter()

    # getrecall()

    # getrecallwrtdetp()