#!/bin/bash

conda activate /eos/user/m/mmcohen/conda_envs/l1AD_env

NTUPLE_FILE='/eos/home-m/mmcohen/ad_trigger_development/ops/data/ntuples/data_dict_20250530_498335.h5'

python l1AD_inference.py $NTUPLE_FILE

conda deactivate

