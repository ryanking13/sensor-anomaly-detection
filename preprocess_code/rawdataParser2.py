# parse csv files, and merge data into one
# divide csv file into parts
# output goes into ./data/ directory
# ./data/aIndex.txt is index file which contains wafer list, and property of wafer
# output is ./data/WAFERNAME.txt, which contains one wafer's data



import csv

#parse date of string format into integer. (0.01ms timescale) 
# don't care change of date
# unit time is 0.01 sec
def dateToInt( string ):
	string = string.replace('-',':')
	string = string.replace(' ',':')
	string = string.replace('.',':')
	#print(string)
	
	split = string.split(':')
	year = int(split[0])
	month = int(split[1])
	day = int(split[2])
	hour = int(split[3])
	minute = int(split[4])
	second = int(split[5])
	csecond = int(split[6])


	p = 1
	totaltime = csecond

	p = p * 100
	totaltime = totaltime + (second * p)

	p = p * 60
	totaltime = totaltime + (minute * p)

	p = p * 60
	totaltime = totaltime + (hour * p)

	p = p * 24
	totaltime = totaltime + (day * p)

	p = p * 31
	totaltime = totaltime + (month * p)

	p = p * 365
	totaltime = totaltime + (year * p)

	return totaltime
	

	
waferNames = set()
waferList = []
waferStart = {}
waferEnd = {}
cnt = 0

# append aIndex.txt
with open('./csv_files/snu_label_2.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter = ',' , quotechar = '|' )
	
	f = open('./data/aIndex.txt','a') #modified
	for row in spamreader:
			
			if row[0] == "material_id" :
				continue
			print("Adding " + row[0] + " " + row[1] + " to aIndex.txt")
			f.write("%s %s\n"%(row[0],row[1]))
			
	f.close()


# get information of each wafer's start & end time
with open('./csv_files/snu_info_2.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter = ',' , quotechar = '|' )
	for row in spamreader:
			if row[0] == "date_time" :
				continue

			waferNames.add(row[4])

			waferinfo = []

			waferinfo.append(row[4]) 				# name
			waferinfo.append( dateToInt(row[0]) )	# time in 0.01 sec format
			waferinfo.append(row[3])				# status
			waferinfo.append(row[8]) 				# stemnum 
			waferList.append(waferinfo)

for wafer in waferList:
	if wafer[0] in waferStart : 
		waferStart[wafer[0]] = min( waferStart[wafer[0]], wafer[1] )
	else :
		waferStart[wafer[0]] = wafer[1]

	if wafer[0] in waferEnd :
		waferEnd[wafer[0]] = max( waferEnd[wafer[0]], wafer[1] )	
	else :
		waferEnd[wafer[0]] = wafer[1]


cnt = 0
ncnt = 0

# make ./data/WAFERNAME.txt files
for name in waferNames:
	cnt = cnt+1
	header = []
	timelist = []
	start = waferStart[name]
	end = waferEnd[name]
	data = {}

	print("Processing data of %s"%name)
	
	with open('./csv_files/snu_data1_2.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',' , quotechar = '|' )
		for row in spamreader:
			
			if row[0] == "date_time" :
				header.append(row[0])
				for i in range(1,43) :
					header.append(row[i])
				continue

			time = dateToInt( row[0] )
			if time < start or time > end:
				continue

			if time not in data :
				timelist.append(time)
				data[time] = []

			for i in range(1,43) :
				data[time].append(row[i])

	with open('./csv_files/snu_data2_2.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',' , quotechar = '|' )
		for row in spamreader:
			if row[0] == "date_time" :
				for i in range(1,42) :
					header.append(row[i])
				continue

			time = dateToInt( row[0] )
			if time < start or time > end:
				continue

			if time not in data :
				data[time] = []

			for i in range(1,42) :
				data[time].append(row[i])

	timelist.sort()
	filename = "./data/" + name + ".txt"
	f = open(filename,'w')

	for x in header:
		f.write("%s "%x)
	f.write("\n")
	
	for time in timelist:
		f.write("%d "%time)
		for x in data[time]:
			f.write("%s "%x)
		f.write("\n")
	f.close()
