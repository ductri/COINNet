#!/bin/bash


CUDA_VISIBLE_DEVICES=1,2 ray start --head --port=6379 --num-cpus=48 --temp-dir=/scratch/tri/tmp/ray/

