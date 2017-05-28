#!/bin/bash

for i in {1..20}
do
	for j in {11..11}
	do
		python3 train_KNN.py $i $j
	done
done
