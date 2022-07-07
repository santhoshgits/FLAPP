import time
import os
import numpy as np
import shutil
import pickle
import math
from collections import Counter, defaultdict
import sys

dire = os.getcwd()

if len(sys.argv) == 3:
	InputFold = sys.argv[1]
	OutFold = sys.argv[2]
else:
	print ('python CreateVectors.py <InputFolder> <OutputFolder> ')	
	sys.exit()
	
	
path_out = dire+'/'+OutFold


if not os.path.exists(path_out):
	os.mkdir(path_out)
else:
	shutil.rmtree(path_out)
	os.mkdir(path_out)





ResidueGrouping = {'GLY':0, 'ALA':0, 'VAL':0, 'ILE':0, 'LEU':0, 'PRO':0, 'ASP':1, 'GLU':1, 'ASN':1, 'GLN':1,
'ARG':2, 'HIS':2, 'LYS':2, 'TYR':3, 'PHE':3, 'TRP':3, 'CYS':4, 'SER':4, 'THR':4, 'MET':4 }


def CenterResidues(arr):
	arr = [ i for i in arr if i[:4] == 'ATOM' and i[74:78].strip() != 'H' ]
	if len(arr) == 0:
		print ('NoResidue')
		sys.exit()
	coord = []
	for i in arr:
		coord.append([i[28:38].strip(), i[38:46].strip(), i[46:54].strip()])
	coord = np.asarray(coord, dtype='float')	
	coord -= np.mean(coord, axis=0)
	dic = {1:"   ", 2:"  ", 3:" ", 4:""}
	brr = []
	dic1 = {}
	for i in range(len(arr)):
		
		if arr[i][13:16]+' '+arr[i][17:26] not in dic1:
			dic1[arr[i][13:16]+' '+arr[i][17:26]] = 0
			j = [ "%.3f"%j for j in coord[i]]
			val = ''.join([ dic[len(k.split(".")[0])]+k for k in j ])
			brr.append(arr[i][:30]+val+arr[i][54:])
	return brr
	
	
def ConstructMatrix(Arr1):
	
	Arr1 = CenterResidues(Arr1)
	# Find uniq Residue Elements
	
	Arr = [ i for i in Arr1 if i[:4] == 'ATOM' and i[74:78].strip() != 'H' and i[13:16].strip() == 'CA']
	
	# Remove Duplicates. This step is essential
	Arr_temp = []
	Dic = {}
	Coord = []
	ResList = []
	for i in Arr:
		var = i[13:16].strip()+" "+i[17:26].strip()
		if var not in Dic:
			Dic[var] = 0
			Arr_temp.append(i)
			Coord.append([i[28:38].strip(), i[38:46].strip(), i[46:54].strip()])
			#ResList.append(i[17:26])
			ResList.append(i[17:20]+'-'+i[21:22]+'-'+i[22:26].strip())
			
	Arr = Arr_temp
	
	
	Coord = np.asarray(Coord, dtype='float')		
	
	
	L = len(Coord)
	DistMatrix = np.zeros((L, L))		
	PCMatrix = np.zeros((L))		
	
	for i in range(L):
		for j in range(L):
			DistMatrix[i][j] = math.sqrt(pow(( Coord[i][0] - Coord[j][0] ),2) + pow(( Coord[i][1] - Coord[j][1] ),2) + pow(( Coord[i][2] - Coord[j][2] ),2))
	
	
	for i in range(len(Arr)):
		PCMatrix[i] = ResidueGrouping[Arr[i][17:20]]
		
		
	temp_dic = { i:0 for i in ResList }
	FullCoord = defaultdict(list)
	FullPDB = defaultdict(list)
	CoordKabsch = []
	AtomKabsch = []
	for i in Arr:
		#print i
		FullCoord[i[17:26]].append([i[28:38].strip(), i[38:46].strip(), i[46:54].strip()])
		FullPDB[i[17:26]].append(i.strip())
		AtomKabsch.append(i)
		if i[13:16].strip() == 'CA':
			CoordKabsch.append([i[28:38].strip(), i[38:46].strip(), i[46:54].strip()])
	CoordKabsch = np.asarray(CoordKabsch, dtype='float')
	
		
	#time.sleep(11)
	
	return DistMatrix, Arr, L, PCMatrix, FullCoord, FullPDB, ResList, CoordKabsch, AtomKabsch
		
 


for i in os.listdir(dire+'/'+InputFold):
	print (i)
	arr = []
	aline = open(dire+'/'+InputFold+'/'+i, 'r').readlines()
	aline = [ line for line in aline if line[:4] == 'ATOM']
	if aline:
		DistMatrix1, Arr1, L1, PCMatrix1, FullCoord1, FullPDB1, ResList1, CoordKabsch1, AtomKabsch1 = ConstructMatrix(aline)
		# DistMatrix1, L1, PCMatrix1
		arr.append(DistMatrix1)
		arr.append(L1)
		arr.append(PCMatrix1)
		arr.append(CoordKabsch1)
		arr.append(ResList1)
		with open(path_out+'/'+i, 'wb') as fp:
			pickle.dump(arr, fp)



'''

for i in os.listdir(path_out):
	with open (path_out+'/'+i, 'rb') as fp:
		arr = pickle.load(fp)
	print arr[0]
	print arr[1]
	print arr[2]
	
	#time.sleep(11)


'''






















