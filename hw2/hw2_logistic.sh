#!/bin/bash 
#python logistic.py --train --train_data_path $3 --train_label_path $4 --test_data_path X_test
python logistic.py --infer --train_data_path $3 --train_label_path $4 --test_data_path $5 --output_dir $6