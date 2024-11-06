#/bin/bash

python src/run_crowdlayer.py data=cifar10n ;\
python src/run_crowdlayer.py data=cifar10n ;\
python src/run_crowdlayer.py data=cifar10n ;\
python src/run_tracereg.py data=cifar10n ;\
python src/run_tracereg.py data=cifar10n ;\
python src/run_tracereg.py data=cifar10n ;\
python src/run_maxmig.py data=cifar10n ;\
python src/run_maxmig.py data=cifar10n ;\
python src/run_maxmig.py data=cifar10n ;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetf;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetf;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetf;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetw;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetw;\
python src/run_geocrowdnet.py data=cifar10n train.lam=1e-2 train=geocrowdnetw;\
# python src/run_meidtm.py data=cifar10n ;\
# python src/run_meidtm.py data=cifar10n ;\
# python src/run_meidtm.py data=cifar10n ;\
# python src/run_ptd.py data=cifar10n ;\
# python src/run_ptd.py data=cifar10n ;\
# python src/run_volminnet.py data=cifar10n ;\
# python src/run_volminnet.py data=cifar10n ;\
# python src/run_volminnet.py data=cifar10n ;\
# python src/run_reweight.py data=cifar10n ;\
# python src/run_reweight.py data=cifar10n ;\
# python src/run_reweight.py data=cifar10n ;\
# python src/run_bltm.py data=cifar10n ;\
# python src/run_bltm.py data=cifar10n ;\
# python src/run_bltm.py data=cifar10n ;\
echo 'Done'

