import time
import os
import sys
import shutil



if len(sys.argv) == 2:
	fold = sys.argv[1]
else:
	print ('Pairs.py <Site-Folder>')
	sys.exit()
	
dire = os.getcwd()
out = open('Pairs.txt', 'w')
for i in os.listdir(dire+'/'+fold):
	#if '-5-' in i:
	for j in os.listdir(dire+'/'+fold):
		#if '_SIA_' in j:
		out.write(i+'\t'+j+'\n')
















