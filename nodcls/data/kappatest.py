import numpy as np 

pp = [0]*4#[111]*4#[0]*4
nn = [0]*4#[106]*4#[0]*4
pn = [0]*4#[12]*4#[0]*4
npp = [0]*4#[47]*4#[0]*4

for fd in [1,2,3,5]:#xrange(1,4,1):
	dctprd = np.load('/media/data1/wentao/tianchi/luna16/CSVFILES/dctptlabel'+str(fd)+'.npy').item()
	for d in xrange(4):
		modprd = np.load('modprd'+str(d+1)+'fd'+str(fd)+'.npy').item()
		for k, v in dctprd.iteritems():
			if v[d] != -1:
				assert v[d] in [0, 1]
				if v[d] == 1 and modprd[k] == 1: pp[d] += 1
				if v[d] == 0 and modprd[k] == 0: nn[d] += 1
				if v[d] == 1 and modprd[k] == 0: pn[d] += 1
				if v[d] == 0 and modprd[k] == 1: npp[d] += 1
print(pp, nn, pn, npp)
for d in xrange(4):
	n = pp[d] + nn[d] + pn[d] + npp[d]
	p0 = (pp[d] + nn[d]) / float(n)
	pe = (pp[d] + pn[d]) * (pp[d] + npp[d])
	pe += (nn[d] + pn[d]) * (nn[d] + npp[d])
	pe /= float(n * n)
	print((p0-pe)/(1-pe))

