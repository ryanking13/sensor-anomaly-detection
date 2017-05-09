# get sensor data from ./data/
# collect sensor data from every wafer
# make aDielist.txt which contains useless sensor lists
# then make reduced wafer data into ./data/reduced/

import numpy as np

waferCnt = 0
sensorNum = 0
sensors = []
sensorVal = []

# collect sum of every wafer
def getSum( waferName, waferIndex ):
	print("[getSum] processing " + waferName + " " + waferIndex )
	global waferCnt
	global sensorNum
	global sensorVal
	global sensors

	waferCnt = waferCnt + 1
	
	lineCnt = 0

	openname = "./data/" + waferName + ".txt"
	f = open(openname,'r')
	lines = f.readlines()

	for line in lines:
		lineCnt = lineCnt+1
		split = line.split()
		
		if waferCnt == 1 and lineCnt == 1 :
			sensorNum = len(split)
			for i in range(0,sensorNum):
				sensors.append(split[i])
				sensorVal.append(0.00)

		if lineCnt == 1 :
			continue

		for i in range(0,sensorNum):
			sensorVal[i] = sensorVal[i] + float(split[i])

	f.close()

# make reduced data file in ./data/reduced
def makeReduced( waferName, waferIndex ):
	print("[makeReduced] processing " + waferName + " " + waferIndex )

	lineCnt = 0

	openname = "./data/" + waferName + ".txt"
	resultname = "./data/reduced/" + waferName + "_reduced.txt"
	infile = open(openname,'r')
	lines = infile.readlines()
	outfile = open(resultname, 'w')

	for line in lines:
		lineCnt = lineCnt+1
		split = line.split()
		
		for i in range(0,sensorNum):
			if sensorVal[i] >= 0.1:
				outfile.write("%s "%split[i])
		outfile.write("\n");

	infile.close()
	outfile.close()
		



indexfile = open("./data/aIndex.txt",'r')
lines = indexfile.readlines()

for line in lines:
	split = line.split()
	getSum(split[0], split[1])

# make aDielist.txt
outfile = open("./data/aDielist.txt",'w')
	
for i in range(0,sensorNum):
	if sensorVal[i] < 0.1:
		outfile.write("%s "%(sensors[i]))
	print( sensors[i] + " " + str(sensorVal[i]) )

outfile.close()
indexfile.close()


# make reduced data in ./data/reduced
indexfile = open("./data/aIndex.txt",'r')
lines = indexfile.readlines()

for line in lines:
	split = line.split()
	makeReduced(split[0],split[1])


