#!/bin/bash

total=`wc -l ${1} | awk '{print $1}'`
dev=$(( total/12 ))
train=$(( total - dev - dev ))
devtail=$(( dev + dev ))
echo "train: ${train} dev: ${dev} test: ${dev}"

datafile=$1
output=$2
shuf ${datafile} > shuf_tmp.txt
head -n ${train} shuf_tmp.txt > $output/twitter_train.txt
tail -n ${devtail} shuf_tmp.txt > shuf_tail.txt
head -n ${dev} shuf_tail.txt > $output/twitter_dev.txt
tail -n ${dev} shuf_tail.txt > $output/twitter_test.txt
rm -f shuf_tmp.txt shuf_tail.txt
