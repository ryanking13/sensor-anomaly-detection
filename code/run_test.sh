#!/bin/bash

for i in {1..10}
do
	for j in {1..25}
	do
		python3 train_LSTM.py $i $j
	done
done
