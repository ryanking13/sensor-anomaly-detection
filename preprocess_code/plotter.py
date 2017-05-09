# Plot graph from data
# data are in ./data/ directory with .txt format
# output plot goes to ./data/plot/ directory with .png format


from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
import numpy as np




def createGraph( waferName, waferIndex ):
	print("processing " + waferName + " " + waferIndex )

	fig = plt.figure(1)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)

	for direction in ["xzero", "yzero"]:
	    ax.axis[direction].set_axisline_style("-|>")
	    ax.axis[direction].set_visible(True)

	for direction in ["left", "right", "bottom", "top"]:
	    ax.axis[direction].set_visible(False)

	x = []
	y = [[]]

	openname = "./data/" + waferName + ".txt"
	f = open(openname,'r')
	lines = f.readlines()

	cnt = 0
	starttime = 0
	sensorNum = 0
	for line in lines:
		cnt = cnt+1
		split = line.split()
		
		if cnt == 1:
			sensorNum = len(split)-1
			for i in range(0,sensorNum):
				y.append([])
			continue
		if cnt == 2:
			starttime = int(split[0])

		curtime = int(split[0])

		x.append(1.0*(curtime-starttime))
		for i in range(1,sensorNum+1):
				y[i].append(float(split[i]))


	# x = np.linspace(-1., 1., 10	)
	for i in range(1,sensorNum+1):
		ax.plot(x, y[i])

	outname = "./data/plot/" + waferIndex + "_" + waferName + ".png"
	plt.savefig(outname)
	#plt.show()


indexfile = open("./data/aIndex.txt",'r')
lines = indexfile.readlines()
for line in lines:
	split = line.split()
	createGraph(split[0], split[1])



