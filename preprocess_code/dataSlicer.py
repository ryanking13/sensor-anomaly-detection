# data slicer
# get reduced data and slice
# modify parameter length to adjust length of data
# every data file has length line
# sliced data files' index goes into bIndex.txt 
import numpy as np

newIndexFile = open('./data/bIndex.txt','w')
length = 1750

def sliceData( waferName, waferIndex ):
	print("[sliceData] processing " + waferName + " " + waferIndex )	
	lineCnt = 0
	openname = "./data/reduced/" + waferName + "_reduced.txt"

	infile = open(openname,'r')
	lines = infile.readlines()
	for line in lines:
		lineCnt = lineCnt+1
	infile.close()

	for i in range(1,100):
		makeData(waferName, waferIndex, i, length, lineCnt)

def makeData( waferName, waferIndex, startIndex , length, lineCnt ):
	if startIndex + length >= lineCnt:
		return 0

	newIndexFile.write("%s %s\n"%(waferName+"_"+str(startIndex),waferIndex))
	outname = "./data/sliced/" + waferName + "_" + str(startIndex) + ".txt"
	openname = "./data/reduced/" + waferName + "_reduced.txt"
	infile = open(openname,'r')
	outfile = open(outname,'w')

	printed = 0
	cnt = 0
	lines = infile.readlines()
	
	for line in lines:
		if cnt == 0:
			outfile.write("%s"%line)
		
		if cnt >= startIndex and cnt < startIndex + length:
			outfile.write("%s"%line)

		cnt = cnt + 1

	infile.close()
	outfile.close()

indexfile = open("./data/aIndex.txt",'r')
lines = indexfile.readlines()

for line in lines:
	split = line.split()
	sliceData(split[0], split[1])

indexfile.close()
newIndexFile.close()