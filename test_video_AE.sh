#!/bin/bash
# testing AE video
python script_testing.py \
    --ModelName AE \
    --ModelSetting Conv3D \
    --Dataset UCSD_P2_256 \
    --ModelRoot ./memae_models/ \
    --DataRoot ./dataset/ \
    --OutRoot ./results/ \
    --Suffix Non