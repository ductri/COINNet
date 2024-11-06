#/bin/bash

python src/run_tracereg.py data=labelme ;\
python src/run_meidtm.py data=labelme ;\
python src/run_meidtm.py data=cifar10n ;\
python src/run_ptd.py data=labelme ;\
python src/run_ptd.py data=cifar10n ;\
echo 'Done'

