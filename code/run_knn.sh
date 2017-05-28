#!/bin/bash

for i in {1..25}
do
	for j in {1..20}
	do
		python3 train_KNN.py $j $i
	done
done
