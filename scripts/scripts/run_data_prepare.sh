#!/bin/bash

scene="doppelherz"
python convert.py -s data/TF-OVOR/${scene}/train_and_test/ --resize --magick_executable convert
python tools/separate_train_test_ply.py --scene ${scene}