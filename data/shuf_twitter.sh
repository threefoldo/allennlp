#!/bin/bash

total=6733
dev=$(( total/12 ))
train=$(( total - dev - dev ))
devtail=$(( dev + dev ))
echo "train: ${train} dev: ${dev} test: ${dev}"

datafile=$1
output=$2
shuf ${datafile} > shuf_tmp.txt
head -n ${train} shuf_tmp.txt > $output/twitter_train.txt
tail -n ${devtail} shuf_tmp.txt | head -n ${dev} > $output/twitter_dev.txt
tail -n ${devtail} shuf_tmp.txt | tail -n ${dev} > $output/twitter_test.txt
