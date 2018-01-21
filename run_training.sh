#!/bin/bash
set -e

# python prepare.py
cd detector
maxeps=150
f=9
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --model res18 -b 64 --resume 064.ckpt --save-dir res18/retrft96$f/ --epochs $maxeps --config config_training$f
for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --model res18 -b 32 --resume results/res18/retrft96$f/00$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 100 ]; then 
	    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --model res18 -b 32 --resume results/res18/retrft96$f/0$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 1000 ]; then
	    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --model res18 -b 32 --resume results/res18/retrft96$f/$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/res18/retrft96$f/val$i/" ]; then
        mkdir results/res18/retrft96$f/val$i/
    fi
    mv results/res18/retrft96$f/bbox/*.npy results/res18/retrft96$f/val$i/
done 