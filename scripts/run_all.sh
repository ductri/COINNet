#!/bin/bash

export CUDA_VISIBLE_DEVICES=0;
python src/run_gce.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_gce.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_gce.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;

python src/run_reweight.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_reweight.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_reweight.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;

python src/run_volminnet.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_volminnet.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_volminnet.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;

python src/run_tracereg.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_tracereg.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_tracereg.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;

python src/run_bltm.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_bltm.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;
python src/run_bltm.py data=imagenet15_3 train.num_epochs=100 train.batch_size=128 train.lr=1e-2;

python src/run_ptd.py data=imagenet15_3 train.batch_size=128 train.lr=1e-2 train.n_epoch_4=200;
python src/run_ptd.py data=imagenet15_3 train.batch_size=128 train.lr=1e-2 train.n_epoch_4=200;
python src/run_ptd.py data=imagenet15_3 train.batch_size=128 train.lr=1e-2 train.n_epoch_4=200;

# Cant custom num_epochs
# XXX errr





