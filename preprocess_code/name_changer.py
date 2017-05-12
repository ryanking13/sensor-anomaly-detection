import os

removal = '_reduced.txt'
files = [f for f in os.listdir('./') if os.path.isfile(os.path.join('./', f))]

for file in files:
        if(file.endswith(removal)):
                    os.rename(file, file[:-len(removal)] + '.txt')
