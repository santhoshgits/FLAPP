import time
import random
import math
import numpy as np
import sys
from collections import Counter, defaultdict
import copy
import re
from numba import jit
from numba import cuda
import numba
import os
import pickle
from multiprocessing import Pool

dire = os.getcwd()


if len(sys.argv) == 5:
	BinFold = sys.argv[1]
	files = sys.argv[2]
	outf = sys.argv[3]
	nprocs = int(sys.argv[4])
else:
	print ('PocketApprox.py <SiteBinaryFolder> <Pairs.txt> <OutFile.txt> <No.of.Cores>') # site1.pdb site2.pdb
	sys.exit()


#pdb_dir = '/home/phd/15/bscsant/pocks/'
#pdb_dir = '/home/nagasuma/Downloads/PDB-70/AlphaFoldPockets/pocks/'

#aline = open(file1, 'r').readlines()
#bline = open(file2, 'r').readlines()	





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
			ResList.append(i[17:26])
			
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
		
 

#DistMatrix1t, Arr1, L1t, PCMatrix1t, FullCoord1, FullPDB1, ResList1, CoordKabsch1t, AtomKabsch1 = ConstructMatrix(aline)
#DistMatrix2t, Arr2, L2t, PCMatrix2t, FullCoord2, FullPDB2, ResList2, CoordKabsch2, AtomKabsch2 = ConstructMatrix(bline)
# Three Predefined Variables
#CoordKabsch1t = np.asarray(CoordKabsch1t, dtype='float')
#CoordKabsch2 = np.asarray(CoordKabsch2, dtype='float')

# DistMatrix1t, L1t, PCMatrix1t, DistMatrix2t, L2t, PCMatrix2t

T = 1.5 # Tau
S = 20

starttime = time.time()



@jit(nopython=True)
def kabsch(P, Q):
	#print(P, Q)
	P_mean = np.zeros(3)
	for i in range(P.shape[1]):
		P_mean[i] = P[:,i].mean()
	P -= P_mean
	
	Q_mean = np.zeros(3)
	for i in range(Q.shape[1]):
		Q_mean[i] = Q[:,i].mean()
	Q -= Q_mean
	
	C = np.dot(np.transpose(P), Q)
	V, S, W = np.linalg.svd(C)
	d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
	if d:
		S[-1] = -S[-1]
		V[:, -1] = -V[:, -1]
	U = np.dot(V, W)
	return U
	




@jit(nopython=True)
def GetAlignments(GlobalSimilarity1, GlobalSimilarity2, CoordKabsch1, CoordKabsch2):
	count_align = -1000
	AlignResTotal = np.zeros((1, 1000,2), dtype='int')-1
	for i in range(len(GlobalSimilarity1)):
	#for i in range(1):
		arr1 = GlobalSimilarity1[i][np.where(GlobalSimilarity1[i] > -1)]
		arr2 = GlobalSimilarity2[i][np.where(GlobalSimilarity2[i] > -1)]

		arr1 = [ int(j) for j in arr1]
		arr2 = [ int(j) for j in arr2]
		
		if len(set(arr1)) <= 3:
			continue
		if len(arr1) != len(arr2):
			continue
		
		arr1_coord = np.zeros((len(arr1), 3))
		arr1_coord_kabsch = np.zeros((len(arr1), 3))
		#print(arr1, arr2)
		for j in range(len(arr1)):
			arr1_coord[j] = CoordKabsch1[arr1[j]]
			arr1_coord_kabsch[j] = CoordKabsch1[arr1[j]]
			
		arr2_coord = np.zeros((len(arr2), 3))
		arr2_coord_kabsch = np.zeros((len(arr2), 3))
		for j in range(len(arr2)):
			arr2_coord[j] = CoordKabsch2[arr2[j]]
			arr2_coord_kabsch[j] = CoordKabsch2[arr2[j]]
		
		
		U = kabsch(arr1_coord_kabsch, arr2_coord_kabsch)
		
		B1 = np.zeros((len(CoordKabsch1),3))
		for j in range(len(CoordKabsch1)):
			B1[j] = CoordKabsch1[j]
		
		
		c1_res = np.zeros(3)
		for j in range(3):
			c1_res[j] = arr1_coord[:,j].mean()
		c2_res = np.zeros(3)
		for j in range(3):
			c2_res[j] = arr2_coord[:,j].mean()	
		
		#print(B1, c1_res, arr1_coord)
		#print(c1_res)
		#print(arr1_coord.mean(axis=0))
		B1 -= c1_res
		B1 = np.dot(B1, U)
		B1 += c2_res
		AlignRes = np.zeros((1000,2), dtype='int')-1
		#print (B1[:3], CoordKabsch2[:3])
		count = 0
		for j in range(len(B1)):
			for k in range(len(CoordKabsch2)):
				ans = math.sqrt(math.pow(( B1[j][0] - CoordKabsch2[k][0] ),2) + math.pow(( B1[j][1] - CoordKabsch2[k][1] ),2) + math.pow(( B1[j][2] - CoordKabsch2[k][2] ),2))
				if ans < 1.0:
					AlignRes[count] = [j,k]
					count += 1
		#print(count)
		if count > count_align:
			count_align = count
			AlignResTotal[0] = AlignRes
	return count_align, AlignResTotal[0]				
		


@jit(nopython=True)
def CheckExtension(S1, S2, P1, P2, DistMatrix1, DistMatrix2):
	
	for i in range(len(S1)):
		if S1[i] != -1:
			#print (S1, S2, P1, P2)
			#print('\n\n')
			
			if abs( DistMatrix1[S1[i]][int(P1)] - DistMatrix2[S2[i]][int(P2)] ) > T:
				return False
	return True



@jit(nopython=True)
def GetCommanElements(DistBin1a, DistBin1b, DistBin1c, DistBin2a, DistBin2b, DistBin2c, DistMatrix1, L1, PCMatrix1, DistMatrix2, L2, PCMatrix2, CoordKabsch1, CoordKabsch2):
	# Residue index, distance of nearby residues, index of nearby residues
	
	#print(DistBin2c)
	count = 0
	Pairs1a = np.zeros(100)
	Pairs1b = np.zeros(100)
	Pairs1a = Pairs1a-1
	Pairs1b = Pairs1b-1
	
	Pairs2a = np.zeros(100)
	Pairs2b = np.zeros(100)
	Pairs2a = Pairs2a-1
	Pairs2b = Pairs2b-1
	
	for i in range(1,len(DistBin1b)):
		for j in range(1,len(DistBin2b)):	
			#print (DistBin1c[i], DistBin2c[j], j, len(DistBin2b))
			if PCMatrix1[DistBin1c[i]] == PCMatrix2[DistBin2c[j]]:
				if DistBin1c[i] > -1 and DistBin2c[j] > -1: # Tag Changed
					if abs(DistBin1b[i] - DistBin2b[j] ) < T:
						#print (i, j,len(DistBin2b),DistBin1a) 
						#time.sleep(1)
						if count >=100:
							break
						Pairs1a[count] = DistBin1a
						Pairs1b[count] = DistBin1c[i]
						Pairs2a[count] = DistBin2a
						Pairs2b[count] = DistBin2c[j] # Tag Changed
						count += 1
	
	
	Pairs1a = Pairs1a[:count]		# res1, res2
	Pairs1b = Pairs1b[:count]	
	Pairs2a = Pairs2a[:count]	
	Pairs2b = Pairs2b[:count]			
	
	#print(Pairs1a)
	
	Pairs = np.zeros((len(Pairs1a),5))
	for i in range(len(Pairs1a)):
		Pairs[i] = [ Pairs1a[i], Pairs1b[i], Pairs2a[i], Pairs2b[i], abs(Pairs1b[i] - Pairs2b[i])   ] # this takes care of order from each residues
	#Pairs = sorted(Pairs, key = lambda x:int(x[4])) # this is not numpy array
	#Pairs = np.array(Pairs, dtype='float')
	ind = np.argsort(Pairs[:,4])
	#print(np.take_along_axis(Pairs, ind, axis=0)) 
	Pairs = Pairs[ind]
	
	PairsHash = np.zeros((len(Pairs),5))
	PairsHashCount = -1
	PairsHashIndexCount = np.zeros(1000, dtype='int')
	PairsHashIndexCount -= 1
	
	for i in range(len(Pairs)):
		
		c = 0
		for j in range(len(PairsHash)):
			
			check = True
			for k in range(len(Pairs[i])):
				if Pairs[i][k] != PairsHash[j][k]:
					check = False
			if check:
				c = 1
		if c == 0:
			PairsHashCount += 1
			PairsHash[PairsHashCount] = Pairs[i]
			PairsHashIndexCount[PairsHashCount] = i		
		
	PairsHashIndexCount = PairsHashIndexCount[np.where(PairsHashIndexCount > -1)]	
	Pairs = Pairs[PairsHashIndexCount]	
	
	
	Pairs1a = np.zeros((len(Pairs),2))
	Pairs2a = np.zeros((len(Pairs),2))
	
	for i in range(len(Pairs)):
		Pairs1a[i] = [ Pairs[i][0], Pairs[i][1] ] 
		Pairs2a[i] = [ Pairs[i][2], Pairs[i][3] ] 
	
	if len(Pairs1a) == 0:
		return 0, np.zeros((1,1), dtype='int')-1
	Answer = np.zeros(len(Pairs1a))
	
	GlobalSimilarity1 = np.zeros((len(Pairs1a),100))
	GlobalSimilarity2 = np.zeros((len(Pairs1a),100))
	Gcount = 0
	
	CutOff = 2
	CutOffCount = 0
	for i in range(len(Pairs1a)):
		#i = 36
		#print(Pairs1a)
		
		Visited1 = np.zeros(30, dtype='int')
		Visited1 = Visited1-1
		Visited1[0] = Pairs1a[i][1]
		
		Visited2 = np.zeros(30, dtype='int')
		Visited2 = Visited2-1
		Visited2[0] = Pairs2a[i][1]
		
		
		Similarity1 = np.zeros(100, dtype='int')
		Similarity2 = np.zeros(100, dtype='int')
		Similarity1 = Similarity1-1
		Similarity2 = Similarity2-1
		Similarity1[0] = Pairs1a[i][0]
		Similarity1[1] = Pairs1a[i][1]
		Similarity2[0] = Pairs2a[i][0]
		Similarity2[1] = Pairs2a[i][1]
		
		CheckCount = 2
		HashCount = 1
		for j in range(len(Pairs2a)):
			#print (Pairs1a[j][1])
			#j = 38
			#if int(Pairs1a[j][1]) not in Visited1 and int(Pairs2a[j][1]) not in Visited2:
				
			if PCMatrix1[int(Pairs1a[j][1])] == PCMatrix2[int(Pairs2a[j][1])]:
				#print('got', int(Pairs1a[j][1]))
				if CheckExtension(Similarity1, Similarity2, Pairs1a[j][1], Pairs2a[j][1], DistMatrix1, DistMatrix2):
					if HashCount > 25:
						break
					if int(Pairs1a[j][1]) not in Visited1:
						Similarity1[CheckCount] = Pairs1a[j][1]
						Similarity2[CheckCount] = Pairs2a[j][1]
						Visited1[HashCount] = int(Pairs1a[j][1])
						Visited2[HashCount] = int(Pairs2a[j][1])
						#print (Visited1, Visited2)
						CheckCount += 1
						HashCount += 1

		Answer[i] = len(np.unique(Similarity1))-1
		if len(np.where(Similarity1 > -1)[0]) > CutOff: # revisit. change to 3
			GlobalSimilarity1[Gcount] = Similarity1
			GlobalSimilarity2[Gcount] = Similarity2
			#print(Similarity1, np.where(Similarity1 > -1)[0], len(np.where(Similarity1 > -1)) )
			Gcount += 1
			CutOffCount += 1
			if CutOffCount > -1:
				CutOff = 3
			
	#print (Answer)
	
	
	#print(GlobalSimilarity1[:3])
	#print(Gcount)
	if Gcount == 0:
		return -1000, np.zeros((1,1), dtype='int')-1
	GlobalSimilarity1 = GlobalSimilarity1[:Gcount]
	GlobalSimilarity2 = GlobalSimilarity2[:Gcount]
	#GIndex = [ i for i in range(len(GlobalSimilarity1)) if len( np.where(GlobalSimilarity1[i] > 0)[0] ) > 1   ] 
	
	#GlobalSimilarity2 = [ GlobalSimilarity2[i] for i in GIndex ]
	#GlobalSimilarity1 = [ GlobalSimilarity1[i] for i in GIndex ]
	#G1 = np.asarray(G1)
	#GlobalSimilarity1 = np.asarray(GlobalSimilarity1)
	#print('\n')
	
	
	Gs = [ sorted(i) for i in GlobalSimilarity1 ]
	#print(Gs)
	Gs = np.asarray(Gs, dtype='int')
	GsZero = np.zeros(Gs.shape, dtype='int')
	
	IndexCount = np.zeros(1000, dtype='int')
	IndexCount -= 1
	
	count = -1
	
	for i in range(len(Gs)):
		c = 1
		#print(i)
		for j in range(len(GsZero)):
			check = True
			for i_k in range(len(Gs[i])):
				if Gs[i][i_k] != GsZero[j][i_k]:
					check = False # no longer same
			if check:
				c = 0
		
		if c == 1:
			#print(i)
			count += 1
			#print(count)
			GsZero[count] = Gs[i]
			IndexCount[count] = i
	
	IndexCount = IndexCount[np.where(IndexCount > -1)]
	

	GlobalSimilarity1 = GlobalSimilarity1[IndexCount]
	GlobalSimilarity2 = GlobalSimilarity2[IndexCount]
	
	#print(GlobalSimilarity1)
	
	
	return GetAlignments(GlobalSimilarity1, GlobalSimilarity2, CoordKabsch1, CoordKabsch2)
	#return max(Answer)	
	#print('\n\n')
			
			

@jit(nopython=True, nogil=True, cache=True)
def Run(DistMatrix1, L1, PCMatrix1, DistMatrix2, L2, PCMatrix2, CoordKabsch1, CoordKabsch2):
	#for n in numba.prange(10000):
	
	
	ResPos1 = np.asarray(list(range(L1)))

	DistBin1a = np.zeros(L1, dtype='int')
	DistBin1a = DistBin1a-1
	DistBin1b = np.zeros((L1,S), dtype='float')
	DistBin1b = DistBin1b-1
	DistBin1c = np.zeros((L1,S), dtype='int')
	DistBin1c = DistBin1c-1
	
	
	for i in range(L1):
		DistStacked = np.stack((DistMatrix1[i],ResPos1), axis=1)
		DistStacked = DistStacked[np.argsort(DistStacked[:,0])][:S]
		DistBin1a[i] = i # Residue index
		for j in range(len(DistStacked)):
			DistBin1b[i][j] = DistStacked[j][0]
			DistBin1c[i][j] = int(DistStacked[j][1])
	
	
	ResPos2 = np.asarray(list(range(L2)))
	#print(L2,'--')
	DistBin2a = [-1]*L2
	DistBin2b = [[-1]*S]*L2
	DistBin2c = [[-1]*S]*L2
	#DistBin2a = np.asarray(DistBin2a, dtype='int')
	DistBin2b = np.asarray(DistBin2b, dtype='float')
	DistBin2c = np.asarray(DistBin2c, dtype='int')
	
	DistBin2a = np.zeros(L2, dtype='int')
	DistBin2a = DistBin2a-1
	DistBin2b = np.zeros((L2,S), dtype='float')
	DistBin2b = DistBin2b-1
	DistBin2c = np.zeros((L2,S), dtype='int')
	DistBin2c = DistBin2c-1
	
	
	for i in range(L2):
		#print(i,' is i',)
		DistStacked = np.stack((DistMatrix2[i],ResPos2), axis=1)
		DistStacked = DistStacked[np.argsort(DistStacked[:,0])][:S]
		DistBin2a[i] = i # Residue index		
		for j in range(len(DistStacked)):
			DistBin2b[i][j] = DistStacked[j][0]
			DistBin2c[i][j] = int(DistStacked[j][1])
			
			
	FinalStore = -1000
	
	
	
	
	AlignResTotal = np.zeros((1, 1000, 2), dtype='int')-1 
	for i in range(len(DistBin1a)):
	#for i in range(1):
		for j in range(0,len(DistBin2a)):
		#for j in range(1):
			#if i == j:
			if PCMatrix1[i] == PCMatrix2[j]:
				#print (i,j, DistBin1a[i])
				MaximumCommon, AlignRes = GetCommanElements(DistBin1a[i], DistBin1b[i], DistBin1c[i], DistBin2a[j], DistBin2b[j], DistBin2c[j],
				DistMatrix1, L1, PCMatrix1, DistMatrix2, L2, PCMatrix2, CoordKabsch1, CoordKabsch2)
				#MaximumCommon = 11
				if MaximumCommon > FinalStore:
					FinalStore = MaximumCommon
					AlignResTotal[0] = AlignRes
				#print(DistBin1b)
				#time.sleep(11)
			
	return FinalStore, AlignResTotal[0]		


'''
aline = open(files,'r').readlines()
out = open('Output','w')
for line in aline:
	line = line.strip()
	site1, site2 = line.split('\t')
	if os.path.exists(dire+'/'+BinFold+'/'+site1) and os.path.exists(dire+'/'+BinFold+'/'+site2):
		stime = time.time()
		with open (dire+'/'+BinFold+'/'+site1, 'rb') as fp:
			arr = pickle.load(fp)
		DistMatrix1 = arr[0]
		L1 = int(arr[1])
		PCMatrix1 = arr[2]
		CoordKabsch1 = arr[3]
		ResList1 = arr[4]
		CoordKabsch1 = np.asarray(CoordKabsch1, dtype='float')
		DistMatrix1 = np.asarray(DistMatrix1, dtype='float')
		PCMatrix1 = np.asarray(PCMatrix1, dtype='float')
		
		print(line)
		with open (dire+'/'+BinFold+'/'+site2, 'rb') as fp:
			arr = pickle.load(fp)
		DistMatrix2 = arr[0]
		L2 = int(arr[1])
		PCMatrix2 = arr[2]
		CoordKabsch2 = arr[3]
		ResList2 = arr[4]
		CoordKabsch2 = np.asarray(CoordKabsch2, dtype='float')
		DistMatrix2 = np.asarray(DistMatrix2, dtype='float')
		PCMatrix2 = np.asarray(PCMatrix2, dtype='float')
		
		
		score, ResCorres = Run(DistMatrix1, L1, PCMatrix1, DistMatrix2, L2, PCMatrix2, CoordKabsch1, CoordKabsch2) 
		ResCorres = [ [ResList1[ResCorres[i][0]], ResList2[ResCorres[i][1]]] for i in range(len(ResCorres)) if ResCorres[i][0] > -1 ]
		if score > 0:
			ResCorres = " ".join([ '_'.join(i) for i in ResCorres ])
		else:
			ResCorres = "NoResidueMatch"
		
		
		etime = time.time()
		print(line, score, ' Time is : ', round(etime-stime,3),'seconds')
		out.write(line+'\t'+str(score)+'\t'+ResCorres+'\t'+str(round(etime-stime,3))+'\n')
out.close()
'''


def BatchRun(PdbPairs):
	site1, site2 = PdbPairs.split('\t')
	if os.path.exists(dire+'/'+BinFold+'/'+site1) and os.path.exists(dire+'/'+BinFold+'/'+site2):
		with open (dire+'/'+BinFold+'/'+site1, 'rb') as fp:
			arr = pickle.load(fp)
		DistMatrix1 = arr[0]
		L1 = int(arr[1])
		PCMatrix1 = arr[2]
		CoordKabsch1 = arr[3]
		ResList1 = arr[4]
		CoordKabsch1 = np.asarray(CoordKabsch1, dtype='float')
		DistMatrix1 = np.asarray(DistMatrix1, dtype='float')
		PCMatrix1 = np.asarray(PCMatrix1, dtype='float')
		
		with open (dire+'/'+BinFold+'/'+site2, 'rb') as fp:
			arr = pickle.load(fp)
		DistMatrix2 = arr[0]
		L2 = int(arr[1])
		PCMatrix2 = arr[2]
		CoordKabsch2 = arr[3]
		ResList2 = arr[4]
		CoordKabsch2 = np.asarray(CoordKabsch2, dtype='float')
		DistMatrix2 = np.asarray(DistMatrix2, dtype='float')
		PCMatrix2 = np.asarray(PCMatrix2, dtype='float')
		
		stime = time.time()
		score, ResCorres = Run(DistMatrix1, L1, PCMatrix1, DistMatrix2, L2, PCMatrix2, CoordKabsch1, CoordKabsch2) 
		etime = time.time()
		ResCorres = [ [ResList1[ResCorres[i][0]], ResList2[ResCorres[i][1]]] for i in range(len(ResCorres)) if ResCorres[i][0] > -1 ]
		if score > 0:
			ResCorres = " ".join([ '_'.join(i) for i in ResCorres ])
		else:
			ResCorres = "NoResidueMatch"
			
		if score < 0:
			score = 0
		score = float(score)	
		minim = sorted( [round(score/L1, 2), round(score/L2, 2)], reverse=True )
		score = list(map(str, [ int(score), L1, L2, minim[0], minim[1] ]))
		score = ' '.join(score)	
			
		return PdbPairs+'\t'+str(score)+'\t'+ResCorres+'\t'+str(round(etime-stime,3))
		
	else:
		return PdbPairs+'\t'+str('0 0 0 0 0')+'\tSiteError\t'+str(round(0,3))



p = Pool(nprocs)

aline = open(files,'r').readlines()


aline = [ line.strip() for line in aline ]
L = len(aline)
#print(aline[:10])

L_local = 0
TimeArr = 0
StartTime = time.time()
print('\n')
print('Compiling the Code. Please Wait...')
print('Utilizing '+str(nprocs)+' CPU Cores....')
Step = 1000
Step = Step*nprocs

out = open(outf,'w')
out.write('Site-1\tSite-2\tNo.Aligned Site-1Residue Site-2Residue F-min F-max\tResidue-Residue Alignments\n')
for i in range(0, L, Step):

	Batch = aline[i:i+Step]
	Result = p.map(BatchRun, Batch)
	L_local += len(Result)
	
	for j in Result:
		#print(j)
		out.write(j+'\n')
		
	EndTime = time.time()
		
	print('Completed {:0.0f} Pairs out of {:0.0f}. Percent Complete : {:3.3f} %. Pairs scanning per second {:0.4f}'.format(L_local, L, (float(L_local)/L)*100, float(L_local)/(EndTime-StartTime)), end='\r')	

print('\n\n')
out.close()







