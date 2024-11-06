#!/bin/bash
python src/run_meidtm.py data=stl10_machine_2_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_2_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_2_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_3_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_3_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_3_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_6_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_6_single_annotator ;\
python src/run_meidtm.py data=stl10_machine_6_single_annotator ;\

python src/run_bltm.py data=stl10_machine_2_single_annotator ;\
python src/run_bltm.py data=stl10_machine_2_single_annotator ;\
python src/run_bltm.py data=stl10_machine_2_single_annotator ;\
python src/run_bltm.py data=stl10_machine_3_single_annotator ;\
python src/run_bltm.py data=stl10_machine_3_single_annotator ;\
python src/run_bltm.py data=stl10_machine_3_single_annotator ;\
python src/run_bltm.py data=stl10_machine_6_single_annotator ;\
python src/run_bltm.py data=stl10_machine_6_single_annotator ;\
python src/run_bltm.py data=stl10_machine_6_single_annotator ;\

python src/run_meidtm.py data=cifar10_machine_6_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_6_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_6_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_6.5_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_6.5_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_6.5_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_7_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_7_single_annotator ;\
python src/run_meidtm.py data=cifar10_machine_7_single_annotator ;\
echo "done"

